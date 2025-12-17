from random import random
from typing import Dict

from corridor import Action, Corridor

class BaseAgent:
    """Interface minimale : implémente select_action(env, obs)."""
    def __init__(self, name: str = "BaseAgent", seed: int | None = None):
        self.name = name
        if seed is not None:
            random.seed(seed)

    def select_action(self, env: Corridor, obs: Dict) -> Action:
        raise NotImplementedError
    
    def run_episode(
        self,
        env: Corridor,
        opponent,
        agent_player: int = 1,
        max_steps: int = 150
    ):
        raise NotImplementedError

    def load(self, path: str):
        raise NotImplementedError

    def save(self, path: str):
        raise NotImplementedError