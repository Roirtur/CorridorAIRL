import numpy as np
import random
from typing import Any, Tuple, List

class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        return self.tree[0]

    def add(self, p: float, data: Any):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx: int, p: float):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, Any]:
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int = 10000, alpha: float = 0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.capacity = capacity
        self.epsilon = 0.01

    def push(self, state, action_features, reward, next_state, done, next_legal_features=None):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = 1.0
        
        data = (state, action_features, reward, next_state, done, next_legal_features)
        self.tree.add(max_p, data)

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Any], np.ndarray, np.ndarray]:
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        is_weights /= is_weights.max()

        return batch, np.array(idxs), np.array(is_weights, dtype=np.float32)

    def update_priorities(self, idxs: np.ndarray, errors: np.ndarray):
        for idx, error in zip(idxs, errors):
            p = (error + self.epsilon) ** self.alpha
            self.tree.update(idx, p)

    def __len__(self):
        return self.tree.n_entries
