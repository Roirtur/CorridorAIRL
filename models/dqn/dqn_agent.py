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
        
        # State dimension: N*N*6 (6 feature planes)
        self.state_dim = board_size * board_size * 6
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
        raise NotImplementedError

    def update(self, state, action_idx, reward, next_state, done, next_legal_actions):
        """Store transition and train network."""
        raise NotImplementedError

    def run_episode(
        self,
        env: Corridor,
        opponent: BaseAgent,
        agent_player: int = 1,
        max_steps: int = 150
    ) -> Dict:
        """Run one training episode."""
        raise NotImplementedError

