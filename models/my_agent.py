from corridor import Corridor, Action
from typing import Dict
import random
from base_agent import BaseAgent

class MyAgent(BaseAgent):
    """Interface minimale : implémente select_action(env, obs)."""
    def __init__(self, name: str = "MyAgent", seed: int | None = None):
        self.name = name
        if seed is not None:
            random.seed(seed)

    def select_action(self, env: Corridor, obs: Dict) -> Action:
        raise NotImplementedError