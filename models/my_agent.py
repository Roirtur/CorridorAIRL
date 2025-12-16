from corridor import Corridor, Action
from typing import Dict
import random
from .dqn_agent import DQNAgent

class MyAgent(DQNAgent):
    """
    MyAgent is a DQN Agent with CNN architecture and Prioritized Experience Replay.
    """
    def __init__(self, name: str = "MyAgent", seed: int | None = None, **kwargs):
        # Initialize with professional defaults
        super().__init__(
            name=name, 
            seed=seed,
            **kwargs
        )
