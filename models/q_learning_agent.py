import random
from typing import Dict
from corridor import Corridor, Action
from .base_agent import BaseAgent
from collections import defaultdict
import numpy as np



def discretize_state_simple(obs: Dict) -> tuple:
    return (
        obs["p1"],
        obs["p2"],
        obs["walls_left"][1],
        obs["walls_left"][2],
    )


class QlearningAgent(BaseAgent):
    """Agent aléatoire : choisit uniformément une action légale."""
    def __init__(
            self,
            gamma, 
            alpha, 
            epsilon, 
            epsilon_decay, 
            min_epsilon, 
            name: str = "QlearningAgent", 
            seed: int | None = None, 
        ):
        super().__init__(name=name, seed=seed)
        self.discretization_fn = discretize_state_simple
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.actions = []
        self.q_table = defaultdict(lambda: defaultdict(float))


    def update(self, state, action, reward, next_state, done):
        state_discrete = self.discretization_fn(state)
        next_state_discrete = self.discretization_fn(next_state)

        target = reward
        if not done:
            target += self.gamma * max(self.q_table[next_state_discrete].values()) if self.q_table[next_state_discrete] else 0.0

        self.q_table[state_discrete][action] += self.alpha * (
            target - self.q_table[state_discrete][action]
        )


    def select_action(self, env: Corridor, obs: Dict) -> Action:
        legal_actions = env.legal_actions()
        discretized_state = self.discretization_fn(obs)

        if np.random.rand() < self.epsilon:
            return random.choice(legal_actions)
        else:
            action_values = []
            for action in legal_actions:
                action_values.append(self.q_table[discretized_state][action])
            best_action_index = np.argmax(action_values)
            return legal_actions[best_action_index]


