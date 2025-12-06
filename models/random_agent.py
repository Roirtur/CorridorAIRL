import random
from typing import Dict
from corridor import Corridor, Action
from .base_agent import BaseAgent

class RandomAgent(BaseAgent):
    """Agent aléatoire : choisit uniformément une action légale."""
    def __init__(self, name: str = "RandomAgent", seed: int | None = None):
        super().__init__(name=name, seed=seed)

    def select_action(self, env: Corridor, obs: Dict) -> Action:
        actions = env.legal_actions()
        if not actions:
            # Devrait être impossible, mais on sécurise
            raise RuntimeError("Aucune action légale disponible.")
        return random.choice(actions)
