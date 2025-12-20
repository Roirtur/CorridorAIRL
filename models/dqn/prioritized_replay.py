
from collections import deque
import random
import numpy as np

class ExperienceReplayPriorized:
    """Prioritized Experience Replay Buffer."""
    def __init__(self, capacity, alpha=0.6):
        """
        alpha is the priority exponent (0 = uniform, 1 = full prioritization)
        """
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = 1e-6
        self.replay_buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done, next_legal_actions=None):
        self.replay_buffer.append((state, action, reward, next_state, done, next_legal_actions))
        self.priorities.append(self.max_priority)

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self))
        
        # priorities to probabilities
        priorities = np.array(self.priorities)
        priorities = priorities ** self.alpha
        probs = priorities / priorities.sum()
        
        indices = np.random.choice(len(self.replay_buffer), size=batch_size, p=probs, replace=False)
        
        samples = [self.replay_buffer[idx] for idx in indices]
        
        return samples, indices
    
    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.replay_buffer)
