from typing import Dict, Optional, Tuple, List
from corridor import Corridor, Action
from models.base_agent import BaseAgent
from models.rl_utils import get_representation_state, state_to_tensor
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim


def action_to_features(action: Action, board_size: int) -> np.ndarray:
    """
    Convert an action to a simple feature vector.
    [is_move, is_wall, r_norm, c_norm, is_horizontal, is_vertical]
    """
    N = float(board_size)
    features = []
    
    kind = action[0]
    if kind == "M":
        _, (r, c) = action
        features = [1.0, 0.0, r/N, c/N, 0.0, 0.0]
    elif kind == "W":
        _, (r, c, ori) = action
        is_h = 1.0 if ori == "H" else 0.0
        is_v = 1.0 if ori == "V" else 0.0
        features = [0.0, 1.0, r/N, c/N, is_h, is_v]
    
    return np.array(features, dtype=np.float32)


class DQN(nn.Module):
    """
    DQN that takes state + action as input and outputs Q-value.
    """
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        input_dim = state_dim + action_dim
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, state, action):
        """
        state: (batch, state_dim)
        action: (batch, action_dim)
        returns: (batch,) Q-values
        """
        x = torch.cat([state, action], dim=1)
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x.squeeze(1)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.queue = deque(maxlen=capacity)

    def push(self, state, action_features, reward, next_state, done, next_legal_actions=None):
        """Store experience."""
        self.queue.append((state, action_features, reward, next_state, done, next_legal_actions))

    def sample(self, batch_size):
        return random.sample(self.queue, batch_size)

    def __len__(self):
        return len(self.queue)


class DQNAgent(BaseAgent):
    """DQN Agent with proper action-value Q-learning"""
    def __init__(
        self,
        name: str = "DQN",
        seed: int | None = None,
        alpha: float = 0.0003,      # Slightly higher for faster learning
        gamma: float = 0.995,       # Higher discount for long-term planning
        epsilon: float = 1.0,       # Start with full exploration
        training_mode: bool = True,
        load_path: Optional[str] = None,
        buffer_size=10000,          # Sufficient buffer
        batch_size=64,              # Smaller batch for faster updates
        target_update_freq=200,     # More frequent target updates
        update_frequency=8,         # Train every N steps (optimized)
        board_size: int = 9,
    ):
        super().__init__(name=name, seed=seed)
        self.training_mode = training_mode
        self.episodes_trained = 0

        # Hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_frequency = update_frequency
        self.learn_step = 0
        self.step_count = 0  # Track total steps
        self.board_size = board_size

        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Updated dimensions
        self.state_dim = 10  # From get_representation_state
        self.action_dim = 6  # From action_to_features

        self.q_network = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=alpha, weight_decay=1e-5)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss for stability

        self.buffer = ReplayBuffer(capacity=buffer_size)

        if load_path:
            pass  # TODO: implement loading

    def _evaluate_actions_batch(self, state_tensor: torch.Tensor, legal_actions: List[Action], network=None) -> Tuple[List[float], Action]:
        """Efficiently evaluate all legal actions in batch."""
        if network is None:
            network = self.q_network
            
        # Prepare batch
        # state_tensor is already (1, state_dim) or (state_dim,)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
            
        # Repeat state for all actions
        num_actions = len(legal_actions)
        state_batch = state_tensor.repeat(num_actions, 1)
        
        action_batch = []
        for action in legal_actions:
            action_features = action_to_features(action, self.board_size)
            action_batch.append(action_features)
        
        action_tensor = torch.FloatTensor(np.array(action_batch)).to(self.device)
        
        with torch.no_grad():
            q_values = network(state_batch, action_tensor).cpu().numpy()
        
        best_idx = np.argmax(q_values)
        return q_values, legal_actions[best_idx]

    def select_action(self, env: Corridor, obs: Dict) -> Action:
        legal_actions = env.legal_actions()
        if not legal_actions:
            return None

        # Epsilon-greedy
        if self.training_mode and np.random.random() < self.epsilon:
            return random.choice(legal_actions)
            
        # Get state features
        state_tuple = get_representation_state(obs, env)
        state_arr = state_to_tensor(state_tuple, self.board_size)
        state_tensor = torch.FloatTensor(state_arr).to(self.device)
        
        # Batch evaluate all legal actions
        _, best_action = self._evaluate_actions_batch(state_tensor, legal_actions)
        
        return best_action

    def update(self, state, action, reward, next_state, next_action, done, env=None, next_legal_actions=None):
        """
        Update DQN.
        state: tuple (unified state)
        next_state: tuple (unified state)
        """
        # Convert state tuple to tensor features
        state_arr = state_to_tensor(state, self.board_size)
        next_state_arr = state_to_tensor(next_state, self.board_size) if next_state else np.zeros(self.state_dim)
        
        action_features = action_to_features(action, self.board_size)
        
        # Store experience
        self.buffer.push(state_arr, action_features, reward, next_state_arr, done, next_legal_actions)
        
        self.step_count += 1

        # Only train every update_frequency steps
        if len(self.buffer) < self.batch_size or self.step_count % self.update_frequency != 0:
            return
        
        batches = self.buffer.sample(self.batch_size)
        
        # Prepare batch tensors
        states = np.array([batch[0] for batch in batches], dtype=np.float32)
        action_features = np.array([batch[1] for batch in batches], dtype=np.float32)
        rewards = np.array([batch[2] for batch in batches], dtype=np.float32)
        next_states = np.array([batch[3] for batch in batches], dtype=np.float32)
        dones = np.array([batch[4] for batch in batches], dtype=np.float32)
        next_legal_actions_list = [batch[5] for batch in batches]
        
        states = torch.FloatTensor(states).to(self.device)
        action_features = torch.FloatTensor(action_features).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        q_values = self.q_network(states, action_features)

        # Target Q-values
        with torch.no_grad():
            next_q_values = []
            for i in range(self.batch_size):
                if dones[i] == 1.0:
                    next_q_values.append(0.0)
                else:
                    next_legal_actions = next_legal_actions_list[i]
                    if next_legal_actions:
                        q_vals, _ = self._evaluate_actions_batch(
                            next_states[i], 
                            next_legal_actions, 
                            network=self.target_network
                        )
                        max_q = np.max(q_vals)
                    else:
                        max_q = 0.0
                    next_q_values.append(max_q)
            
            next_q_values = torch.FloatTensor(next_q_values).to(self.device)
            target = rewards + self.gamma * next_q_values * (1 - dones)
        
        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())


