from typing import Dict, Optional, Tuple, List
from corridor import Corridor, Action
from models.base_agent import BaseAgent
from models.rl_utils import get_canonical_state
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim


def state_to_features(obs: Dict, board_size: int) -> np.ndarray:
    """
    Convert observation to feature vector for DQN.
    Uses more strategic features for better learning.
    """
    N = board_size
    features = []
    
    # Current player perspective
    to_play = obs['to_play']
    my_pos = obs['p1'] if to_play == 1 else obs['p2']
    opp_pos = obs['p2'] if to_play == 1 else obs['p1']
    my_goal = N - 1 if to_play == 1 else 0
    opp_goal = 0 if to_play == 1 else N - 1
    
    # Normalized positions
    features.append(my_pos[0] / N)
    features.append(my_pos[1] / N)
    features.append(opp_pos[0] / N)
    features.append(opp_pos[1] / N)
    
    # Distance to goal (key strategic feature)
    my_dist = abs(my_pos[0] - my_goal)
    opp_dist = abs(opp_pos[0] - opp_goal)
    features.append(my_dist / N)
    features.append(opp_dist / N)
    
    # Relative advantage
    features.append((opp_dist - my_dist) / N)
    
    # Walls remaining
    my_walls = obs['walls_left'][to_play]
    opp_walls = obs['walls_left'][3 - to_play]
    features.append(my_walls / 10)
    features.append(opp_walls / 10)
    
    # Wall advantage
    features.append((my_walls - opp_walls) / 10)
    
    # Manhattan distance between players
    manhattan = abs(my_pos[0] - opp_pos[0]) + abs(my_pos[1] - opp_pos[1])
    features.append(manhattan / (2 * N))
    
    # Number of moves available (mobility)
    move_count = 0
    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
        nr, nc = my_pos[0] + dr, my_pos[1] + dc
        if 0 <= nr < N and 0 <= nc < N:
            move_count += 1
    features.append(move_count / 4)
    
    return np.array(features, dtype=np.float32)


def action_to_features(action: Action, obs: Dict, board_size: int) -> np.ndarray:
    """
    Convert an action to a feature vector with strategic information.
    """
    N = board_size
    features = []
    
    to_play = obs['to_play']
    my_pos = obs['p1'] if to_play == 1 else obs['p2']
    opp_pos = obs['p2'] if to_play == 1 else obs['p1']
    my_goal = N - 1 if to_play == 1 else 0
    
    kind = action[0]
    if kind == "M":
        _, (r, c) = action
        features.append(1.0)  # Is move
        features.append(0.0)  # Not wall
        features.append(r / N)
        features.append(c / N)
        features.append(0.0)  # Wall orientation
        
        # Distance change to goal
        old_dist = abs(my_pos[0] - my_goal)
        new_dist = abs(r - my_goal)
        features.append((old_dist - new_dist) / N)  # Positive if moving closer
        
        # Distance to opponent
        dist_to_opp = abs(r - opp_pos[0]) + abs(c - opp_pos[1])
        features.append(dist_to_opp / (2 * N))
        
    elif kind == "W":
        _, (r, c, ori) = action
        features.append(0.0)  # Not move
        features.append(1.0)  # Is wall
        features.append(r / N)
        features.append(c / N)
        features.append(1.0 if ori == "H" else 0.0)
        
        # Wall blocks opponent (approximate)
        wall_blocks_opp = 1.0 if abs(r - opp_pos[0]) <= 1 and abs(c - opp_pos[1]) <= 1 else 0.0
        features.append(wall_blocks_opp)
        features.append(0.0)  # Placeholder
    
    return np.array(features, dtype=np.float32)


class DQN(nn.Module):
    """
    DQN that takes state + action as input and outputs Q-value.
    Optimized for speed with smaller architecture.
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

    def push(self, state, action_features, reward, next_state, next_obs, done, next_legal_actions=None):
        """Store experience including next observation and legal actions for proper target computation."""
        self.queue.append((state, action_features, reward, next_state, next_obs, done, next_legal_actions))

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
        self.state_dim = 12  # From improved state_to_features
        self.action_dim = 7  # From improved action_to_features

        self.q_network = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=alpha, weight_decay=1e-5)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss for stability

        self.buffer = ReplayBuffer(capacity=buffer_size)

        if load_path:
            pass  # TODO: implement loading

    def _evaluate_actions_batch(self, state_features: np.ndarray, legal_actions: List[Action], obs: Dict, network=None) -> Tuple[List[float], Action]:
        """Efficiently evaluate all legal actions in batch."""
        if network is None:
            network = self.q_network
            
        # Prepare batch
        state_batch = []
        action_batch = []
        
        for action in legal_actions:
            action_features = action_to_features(action, obs, self.board_size)
            state_batch.append(state_features)
            action_batch.append(action_features)
        
        state_tensor = torch.FloatTensor(np.array(state_batch)).to(self.device)
        action_tensor = torch.FloatTensor(np.array(action_batch)).to(self.device)
        
        with torch.no_grad():
            q_values = network(state_tensor, action_tensor).cpu().numpy()
        
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
        state_features = state_to_features(obs, self.board_size)
        
        # Batch evaluate all legal actions
        _, best_action = self._evaluate_actions_batch(state_features, legal_actions, obs)
        
        return best_action

    def update(self, state, action, reward, next_state, next_action, done, env=None, next_obs=None, obs=None, next_legal_actions=None):
        # Convert state tuple to features if needed
        if isinstance(state, tuple):
            return
        
        # Get action features using proper observation
        if obs is None:
            # Fallback: reconstruct basic obs from state (not ideal but works)
            obs = {'to_play': 1, 'p1': (int(state[0] * self.board_size), int(state[1] * self.board_size)), 
                   'p2': (int(state[2] * self.board_size), int(state[3] * self.board_size))}
        
        action_features = action_to_features(action, obs, self.board_size)
        
        # Store experience with next observation and legal actions
        self.buffer.push(state, action_features, reward, next_state, next_obs, done, next_legal_actions)
        
        self.step_count += 1

        # Only train every update_frequency steps
        if len(self.buffer) < self.batch_size or self.step_count % self.update_frequency != 0:
            return
        
        batches = self.buffer.sample(self.batch_size)
        
        # Prepare batch tensors
        states = np.array([batch[0] for batch in batches], dtype=np.float32)
        action_features = np.array([batch[1] for batch in batches], dtype=np.float32)
        rewards = np.array([batch[2] for batch in batches], dtype=np.float32)
        next_states = np.array([batch[3] if batch[3] is not None else batch[0] for batch in batches], dtype=np.float32)
        next_obs_list = [batch[4] for batch in batches]
        dones = np.array([batch[5] for batch in batches], dtype=np.float32)
        next_legal_actions_list = [batch[6] if len(batch) > 6 else None for batch in batches]
        
        states = torch.FloatTensor(states).to(self.device)
        action_features = torch.FloatTensor(action_features).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        q_values = self.q_network(states, action_features)

        # Target Q-values - properly compute max over legal actions
        with torch.no_grad():
            next_q_values = []
            for i, next_obs in enumerate(next_obs_list):
                if next_obs is None or dones[i] == 1.0:
                    # Terminal state
                    next_q_values.append(0.0)
                else:
                    # Use stored legal actions if available
                    next_legal_actions = next_legal_actions_list[i]
                    if next_legal_actions and len(next_legal_actions) > 0:
                        # Batch evaluate legal actions for this next state
                        q_vals, _ = self._evaluate_actions_batch(
                            next_states[i].cpu().numpy(), 
                            next_legal_actions, 
                            next_obs,
                            network=self.target_network
                        )
                        max_q = np.max(q_vals)
                    else:
                        # Fallback: estimate with 0 (conservative)
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


