from collections import defaultdict

import numpy as np


class QLearningAgent:
    def __init__(
        self, env, discretization_fn, gamma, alpha, epsilon, epsilon_decay, min_epsilon
    ):
        self.env = env
        self.discretization_fn = discretization_fn
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.actions = range(env.action_space.n)
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))

    def play(self, state):
        state_discrete = self.discretization_fn(state)
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return int(np.argmax(self.q_table[state_discrete]))

    def update(self, state, action, reward, next_state, done):
        state_discrete = self.discretization_fn(state)
        next_state_discrete = self.discretization_fn(next_state)

        target = reward
        if not done:
            target += self.gamma * np.max(self.q_table[next_state_discrete])

        self.q_table[state_discrete][action] += self.alpha * (
            target - self.q_table[state_discrete][action]
        )
        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)