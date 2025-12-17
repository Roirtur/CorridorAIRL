from corridor import Corridor, Action
from typing import Dict, Optional
import random
from collections import defaultdict

from models.base_agent import BaseAgent
from utils.representation import tabular_state_representation

class QlearningAgent(BaseAgent):
    """Q-learning Agent"""
    def __init__(
        self, 
        min_epsilon: float = 0.01,
        name: str = "Q-learn", 
        seed: int | None = None,
        alpha: float = 0.05,
        gamma: float = 0.995,
        epsilon: float = 0.1,
        training_mode: bool = True,
        load_path: Optional[str] = None,
    ):
        super().__init__(name=name, seed=seed)
        
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        
        self.q_table = defaultdict(float)
        
        self.training_mode = training_mode
        
        if load_path:
            self.load(load_path)
            
        if not self.training_mode:
            self.epsilon = 0.0

    def select_action(self, env: Corridor, obs: Dict) -> Action:
        legal_actions = env.legal_actions()
        if not legal_actions:
            return None

        if self.training_mode and random.random() < self.epsilon:
            return random.choice(legal_actions)
            
        # normalized state
        state_features = tabular_state_representation(obs, env)

        # choose action with highest Q-value
        q_values = [self.q_table[(state_features, action)] for action in legal_actions]
        max_q = max(q_values)
        best_actions = [action for action, q in zip(legal_actions, q_values) if q == max_q]
        
        return best_actions[0]

    def update(self, state, action, reward, next_state, next_legal_actions, done):
        current_q = self.q_table[(state, action)]
        
        if done or next_state is None or next_legal_actions is None:
            target = reward
        else:
            next_q_values = [self.q_table[(next_state, a)] for a in next_legal_actions]
            target = reward + self.gamma * max(next_q_values) if next_q_values else reward
        
        self.q_table[(state, action)] += self.alpha * (target - current_q)