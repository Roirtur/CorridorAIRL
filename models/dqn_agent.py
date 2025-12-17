from typing import Dict, Optional, Tuple, List, Any
from corridor import Corridor, Action
from models.base_agent import BaseAgent
from models.utils.persistence import load_model
from models.utils.features import get_grid_state, flip_action, action_to_features
from models.dqn.buffer import PrioritizedReplayBuffer
from models.dqn.network import DuelingDQN
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

class DQNAgent(BaseAgent):
    def __init__(
        self,
        name: str = "DQN_PER",
        seed: int | None = None,
        alpha: float = 0.0001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        training_mode: bool = True,
        load_path: Optional[str] = None,
        buffer_size: int = 20000,
        batch_size: int = 64,
        target_update_freq: int = 500,
        update_frequency: int = 4,
        board_size: int = 9,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4
    ):
        super().__init__(name=name, seed=seed)
        self.training_mode = training_mode
        self.episodes_trained = 0

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_frequency = update_frequency
        self.step_count = 0
        self.board_size = board_size
        
        self.per_beta = per_beta_start
        self.per_beta_increment = (1.0 - per_beta_start) / 10000

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dimensions: 4 grids (MyPos, OppPos, H, V)
        self.input_channels = 4
        self.action_dim = 6

        self.q_network = DuelingDQN(self.input_channels, self.action_dim, board_size).to(self.device)
        self.target_network = DuelingDQN(self.input_channels, self.action_dim, board_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=alpha)
        
        self.buffer = PrioritizedReplayBuffer(capacity=buffer_size, alpha=per_alpha)

        if load_path:
            self.load(load_path)

    def preprocess_state(self, env: Corridor, obs: Dict) -> Tuple[np.ndarray, bool]:
        """
        Returns (state_tensor, is_flipped)
        state_tensor is (4, N, N) grid.
        """
        return get_grid_state(obs, env)

    def approximate_q_value(self, state: np.ndarray, action: Action) -> float:
        # Note: state here is expected to be the tensor part only
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_features = action_to_features(action, self.board_size)
        action_tensor = torch.FloatTensor(action_features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_value = self.q_network(state_tensor, action_tensor).item()
        return q_value

    def _evaluate_actions_batch(self, state_tensor: torch.Tensor, action_features_list: List[np.ndarray], network=None) -> Tuple[np.ndarray, int]:
        if network is None:
            network = self.q_network
            
        if state_tensor.dim() == 3:
            state_tensor = state_tensor.unsqueeze(0)
            
        num_actions = len(action_features_list)
        state_batch = state_tensor.repeat(num_actions, 1, 1, 1)
        
        action_tensor = torch.FloatTensor(np.array(action_features_list)).to(self.device)
        
        with torch.no_grad():
            q_values = network(state_batch, action_tensor).cpu().numpy()
        
        best_idx = np.argmax(q_values)
        return q_values, best_idx

    def select_action(self, env: Corridor, obs: Dict) -> Action:
        legal_actions = env.legal_actions()
        if not legal_actions:
            return None

        if self.training_mode and np.random.random() < self.epsilon:
            return random.choice(legal_actions)
            
        state_tensor_np, is_flipped = self.preprocess_state(env, obs)
        state_tensor = torch.FloatTensor(state_tensor_np).unsqueeze(0).to(self.device)
        
        # Efficient single-pass evaluation
        canonical_features_list = []
        for act in legal_actions:
            can_act = flip_action(act, self.board_size) if is_flipped else act
            canonical_features_list.append(action_to_features(can_act, self.board_size))
        
        action_tensor = torch.FloatTensor(np.array(canonical_features_list)).to(self.device)
        state_batch = state_tensor.repeat(len(legal_actions), 1, 1, 1)
        
        with torch.no_grad():
            q_values = self.q_network(state_batch, action_tensor).cpu().numpy()
        
        best_idx = np.argmax(q_values)
        return legal_actions[best_idx]

    def update(self, state, action, reward, next_state, next_action, done, env=None, next_legal_actions=None):
        state_tensor, is_flipped = state
        
        # Canonical action features
        can_action = flip_action(action, self.board_size) if is_flipped else action
        action_features = action_to_features(can_action, self.board_size)
        
        if next_state:
            next_state_tensor, next_is_flipped = next_state
        else:
            next_state_tensor = np.zeros_like(state_tensor)
            next_is_flipped = False
            
        # Canonical next legal actions
        next_legal_features = []
        if next_legal_actions:
            for act in next_legal_actions:
                c_act = flip_action(act, self.board_size) if next_is_flipped else act
                next_legal_features.append(action_to_features(c_act, self.board_size))
                
        self.buffer.push(state_tensor, action_features, reward, next_state_tensor, done, next_legal_features)
        
        self.step_count += 1
        self.per_beta = min(1.0, self.per_beta + self.per_beta_increment)

        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        if len(self.buffer) < self.batch_size or self.step_count % self.update_frequency != 0:
            return
        
        batches, idxs, is_weights = self.buffer.sample(self.batch_size, self.per_beta)
        
        states = np.array([b[0] for b in batches], dtype=np.float32)
        actions = np.array([b[1] for b in batches], dtype=np.float32)
        rewards = np.array([b[2] for b in batches], dtype=np.float32)
        next_states = np.array([b[3] for b in batches], dtype=np.float32)
        dones = np.array([b[4] for b in batches], dtype=np.float32)
        
        state_t = torch.FloatTensor(states).to(self.device)
        action_t = torch.FloatTensor(actions).to(self.device)
        reward_t = torch.FloatTensor(rewards).to(self.device)
        next_state_t = torch.FloatTensor(next_states).to(self.device)
        done_t = torch.FloatTensor(dones).to(self.device)
        weights_t = torch.FloatTensor(is_weights).to(self.device)

        curr_q = self.q_network(state_t, action_t)

        # Vectorized next_q calculation
        non_terminal_idxs = []
        all_next_states = []
        all_next_actions = []
        action_counts = []
        
        for i in range(self.batch_size):
            if dones[i]:
                continue
            
            nl_features = batches[i][5]
            if not nl_features:
                continue
                
            count = len(nl_features)
            if count == 0:
                continue

            non_terminal_idxs.append(i)
            # Repeat state 'count' times. next_state_t[i] is (C, H, W) -> (count, C, H, W)
            all_next_states.append(next_state_t[i].unsqueeze(0).repeat(count, 1, 1, 1)) 
            all_next_actions.append(np.array(nl_features))
            action_counts.append(count)

        next_q_values = torch.zeros(self.batch_size, device=self.device)
        
        if non_terminal_idxs:
            flat_next_states = torch.cat(all_next_states, dim=0)
            flat_next_actions = torch.FloatTensor(np.concatenate(all_next_actions, axis=0)).to(self.device)
            
            with torch.no_grad():
                flat_q_values = self.target_network(flat_next_states, flat_next_actions)
            
            # Split back to samples and take max
            per_sample_q = torch.split(flat_q_values, action_counts)
            max_q_list = [q.max() for q in per_sample_q]
            
            for idx, max_q in zip(non_terminal_idxs, max_q_list):
                next_q_values[idx] = max_q
            
        next_q_t = next_q_values
        target_q = reward_t + (1 - done_t) * self.gamma * next_q_t
        
        loss = (weights_t * (curr_q - target_q) ** 2).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        errors = torch.abs(curr_q - target_q).detach().cpu().numpy()
        self.buffer.update_priorities(idxs, errors)

    def load(self, path: str):
        load_model(self, path)


