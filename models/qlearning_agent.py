from corridor import Corridor, Action
from typing import Dict, Optional
import random
from .base_agent import BaseAgent
import numpy as np
from collections import defaultdict
from .rl_utils import get_representation_state

class QlearningAgent(BaseAgent):
    """Q-learning Agent"""
    def __init__(
        self, 
        min_epsilon: float = 0.01,
        name: str = "Q-learn", 
        seed: int | None = None,
        alpha: float = 0.05,      # Learning rate
        gamma: float = 0.995,    # Discount factor
        epsilon: float = 0.1,    # Exploration rate
        training_mode: bool = True,
        load_path: Optional[str] = None,
    ):
        super().__init__(name=name, seed=seed)
        
        # Hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        
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
        Q-Learning update: Q(S, A) <- Q(S, A) + alpha * [R + gamma * max_a Q(S', a') - Q(S, A)]
        Note: state and next_state are already canonical tuples from rl_utils.py
        Note: next_action is ignored for Q-learning (used in SARSA), kept for API consistency
        """
        current_q = self.q_table[(state, action)]
        
        if done or next_state is None:
            target = reward
        else:
            # Compute max over legal actions only
            if next_legal_actions is None:
                target = reward
            else:
                # Proper Q-learning: max over legal actions
                next_q_values = [self.q_table[(next_state, a)] for a in next_legal_actions]
                target = reward + self.gamma * max(next_q_values) if next_q_values else reward
        
        self.q_table[(state, action)] += self.alpha * (target - current_q)