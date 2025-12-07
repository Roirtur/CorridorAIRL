from corridor import Corridor, Action
from typing import Dict, Optional, Any
import random

class BaseAgent:
    """Interface minimale : implémente select_action(env, obs)."""
    def __init__(self, name: str = "BaseAgent", seed: int | None = None):
        self.name = name
        if seed is not None:
            random.seed(seed)

    def select_action(self, env: Corridor, obs: Dict) -> Action:
        raise NotImplementedError

    def update(self, state, action, reward, next_state, next_action, done):
        """
        Update the agent's knowledge.
        """
        pass

    def train(self, env: Corridor, adversary: 'BaseAgent', num_episodes: int, save_path: Optional[str] = None):
        """
        Train the agent against an adversary using the shared training loop.
        """
        from .rl_utils import train_loop
        train_loop(self, env, adversary, num_episodes, save_path)

    def save(self, path: str, env: Corridor, adversary: 'BaseAgent'):
        """
        Save the agent's model.
        """
        from .rl_utils import save_model
        # Assuming the agent tracks episodes_trained, otherwise 0
        episodes = getattr(self, 'episodes_trained', 0)
        save_model(self, path, env, adversary.name, episodes)

    def load(self, path: str):
        """
        Load the agent's model.
        """
        from .rl_utils import load_model
        load_model(self, path)