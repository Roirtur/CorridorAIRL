from corridor import Corridor, Action
from typing import Dict, List, Optional
import random
from .base_agent import BaseAgent
import numpy as np
from collections import defaultdict
from .rl_utils import get_representation_state

class SarsaAgent(BaseAgent):
    """Sarsa Agent"""
    def __init__(
        self, 
        name: str = "Sarsa", 
        seed: int | None = None,
        alpha: float = 0.1,      # Learning rate
        gamma: float = 0.995,    # Discount factor
        epsilon: float = 0.1,    # Exploration rate
        training_mode: bool = True,
        load_path: Optional[str] = None
    ):
        super().__init__(name=name, seed=seed)
        
        # Hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Q-table: maps (state_features, action) -> Q-value
        self.q_table = defaultdict(float)
        
        # For tracking
        self.training_mode = training_mode
        self.episodes_trained = 0
        
        if load_path:
            self.load(load_path)
            
        if not self.training_mode:
            self.epsilon = 0.0 # Greedy in evaluation

    def select_action(self, env: Corridor, obs: Dict) -> Action:
        legal_actions = env.legal_actions()
        if not legal_actions:
            return None

        # Epsilon-greedy
        if self.training_mode and np.random.random() < self.epsilon:
            return random.choice(legal_actions)
            
        # Get normalized state
        state_features = get_representation_state(obs, env)

        # Greedy exploitation: choose action with highest Q-value
        best_q = -float('inf')
        best_actions = []
        
        for action in legal_actions:
            q_val = self.q_table[(state_features, action)]
            if q_val > best_q:
                best_q = q_val
                best_actions = [action]
            elif q_val == best_q:
                best_actions.append(action)
        
        if not best_actions:
            return random.choice(legal_actions)
            
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, next_action, done, env=None, next_legal_actions=None):
        """
        SARSA update: Q(S, A) <- Q(S, A) + alpha * [R + gamma * Q(S', A') - Q(S, A)]
        """
        current_q = self.q_table[(state, action)]
        
        if done:
            target = reward
        else:
            next_q = self.q_table[(next_state, next_action)]
            target = reward + self.gamma * next_q
            
        self.q_table[(state, action)] += self.alpha * (target - current_q)

        