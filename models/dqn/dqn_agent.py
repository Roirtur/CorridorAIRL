import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional
import random

from corridor import Corridor, Action
from models import BaseAgent
from utils.representation import approximation_agent_state_representation
from .dqn_network import DQN
from .prioritized_replay import ExperienceReplayPriorized
from .action_encoder import ActionEncoder


class DQNAgent(BaseAgent):
    def __init__(
        self,
        name: str = "DQN",
        seed: int | None = None,
        board_size: int = 9,
        gamma: float = 0.99,
        alpha: float = 0.001,
        epsilon: float = 0.5,
        epsilon_decay: float = 0.999,
        min_epsilon: float = 0.01,
        buffer_size: int = 5000,
        batch_size: int = 32,
        target_update_freq: int = 100,
        training_mode: bool = True,
        load_path: Optional[str] = None,
    ):
        super().__init__(name=name, seed=seed)
        
        self.board_size = board_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.training_mode = training_mode

        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learn_step = 0

        # Action encoder for Corridor actions
        self.action_encoder = ActionEncoder(board_size=board_size)
        
        # State shape: (6, N, N) - 6 feature planes in channels-first format
        self.state_shape = (6, board_size, board_size)
        self.action_dim = self.action_encoder.action_space_size

        # Device selection (CUDA if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DQN Agent using device: {self.device}")

        # Neural networks (CNN-based Dueling architecture)
        self.q_network = DQN(board_size, self.action_dim).to(self.device)
        self.target_network = DQN(board_size, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer and loss (Huber loss is more stable than MSE)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=alpha, eps=1e-4)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

        # Replay buffer
        self.buffer = ExperienceReplayPriorized(capacity=buffer_size)
        
        if load_path:
            self.load(load_path)
            
        if not self.training_mode:
            self.epsilon = 0.0

    def select_action(self, env: Corridor, obs: Dict) -> Action:
        """Select action following BaseAgent interface with advanced exploration."""
        legal_actions = env.legal_actions()
        if not legal_actions:
            return None

        if self.training_mode and np.random.random() < self.epsilon:
            action = random.choice(legal_actions)
            return action
        
        discretized_state = approximation_agent_state_representation(obs)
        state_tensor = torch.tensor(discretized_state, dtype=torch.float32).to(self.device).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        legal_actions_indexes = self.action_encoder.encode_legal_actions(legal_actions)
        
        legal_q = q_values[0, legal_actions_indexes]
        best_legal_idx_in_subset = torch.argmax(legal_q).item()
        best_legal_action_index = legal_actions_indexes[best_legal_idx_in_subset]
        
        return self.action_encoder.decode(best_legal_action_index)

    def update(self, state, action_idx, reward, next_state, done, next_legal_actions):
        """Store transition and train network."""
        
        self.buffer.push(state, action_idx, reward, next_state, done, next_legal_actions)

        if len(self.buffer) < self.batch_size:
            return
        
        batch, indices = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones, next_legal_actions_batch = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_values_all = self.q_network(states)
        current_q_values = q_values_all.gather(1, actions.unsqueeze(1))

        batch_size = states.shape[0]
        legal_mask = torch.zeros(batch_size, self.action_dim, dtype=torch.bool, device=self.device)

        for i, legal_acts in enumerate(next_legal_actions_batch):
            if not dones[i] and legal_acts:
                legal_indices = self.action_encoder.encode_legal_actions(legal_acts)
                legal_mask[i, legal_indices] = True

        with torch.no_grad():
            next_q_values_online = self.q_network(next_states)
            
            masked_q_values = torch.where(
                legal_mask,
                next_q_values_online,
                torch.tensor(-1e9, device=self.device)
            )
            
            best_next_actions = masked_q_values.argmax(dim=1)

            next_q_values_target = self.target_network(next_states)
            next_q_values = next_q_values_target.gather(1, best_next_actions.unsqueeze(1)).squeeze(1)

            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            target_q_values = target_q_values.unsqueeze(1)

        loss = self.loss_fn(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)

        self.optimizer.step()

        td_errors = (target_q_values.squeeze() - current_q_values.squeeze()).abs().detach().cpu().numpy()
        self.buffer.update_priorities(indices, td_errors)

        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def run_episode(
        self,
        env: Corridor,
        opponent: BaseAgent,
        agent_player: int = 1,
        max_steps: int = 100
    ) -> Dict:
        obs = env.reset()
        
        episode_reward = .0
        steps = 0

        prev_state = None
        prev_action_idx = None
        prev_reward = .0

        while steps < max_steps:
            current_player = obs["to_play"]
            
            if current_player == agent_player:
                agent = self
                is_learning = True
            else:
                agent = opponent
                is_learning = False

            current_legal_actions = env.legal_actions()
            current_state = approximation_agent_state_representation(obs)
            
            action = agent.select_action(env, obs)
            
            # execute action
            next_obs, _, done, info = env.step(action)
            
            if is_learning:
                
                next_reward = -0.01

                action_idx = self.action_encoder.encode(action)
                
                if prev_state is not None:
                    self.update(
                        prev_state,
                        prev_action_idx,
                        prev_reward,
                        current_state,
                        False,
                        current_legal_actions
                    )
                
                episode_reward += next_reward
                
                prev_state = current_state
                prev_action_idx = action_idx
                prev_reward = next_reward

            if done:
                if prev_state is not None:
                    winner = info.get("winner")
                    if winner == agent_player:
                        final_reward = 1.0
                    elif winner is None:
                        final_reward = -0.5
                    else:
                        final_reward = -1.0
                    
                    final_state_rep = approximation_agent_state_representation(next_obs)
                    
                    self.update(
                        prev_state,
                        prev_action_idx,
                        final_reward,
                        final_state_rep,
                        True,
                        []
                    )
                    
                    episode_reward += final_reward
                break
            
            obs = next_obs
            steps += 1
        
        return {
            "reward": episode_reward,
            "steps": steps
        }

    def save(self, path: str):
        """Save NN weights to file."""
        from utils.saving import save_approximation_agent_model
        save_approximation_agent_model(self, path)

    def load(self, path: str):
        """Load NN weights from file."""
        from utils.saving import load_approximation_agent_model
        load_approximation_agent_model(self, path)