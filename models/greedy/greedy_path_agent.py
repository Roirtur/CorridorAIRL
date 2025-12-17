import random
from typing import Dict, Tuple
from corridor import Corridor, Action
from models.base_agent import BaseAgent

class GreedyPathAgent(BaseAgent):
    """
    Heuristique: privilégie les déplacements qui rapprochent le pion de sa ligne but.
    Ne place des murs que très rarement (ou jamais).
    """
    def __init__(self, name: str = "GreedyPathAgent", wall_prob: float = 0.0, seed: int | None = None):
        super().__init__(name=name, seed=seed)
        self.wall_prob = wall_prob

    def select_action(self, env: Corridor, obs: Dict) -> Action:
        actions = env.legal_actions()
        # Filtrer les déplacements
        move_actions = [(a, dst) for (a, dst) in actions if a == "M"]
        if not move_actions:
            # Si aucun déplacement légal, choisir un mur légal (si présent)
            return random.choice(actions)

        me = 1 if obs["to_play"] == 1 else 2
        target_row = env.N - 1 if me == 1 else 0

        # Choisir le move qui minimise la distance (en ligne) vers la ligne cible
        def score_move(dst: Tuple[int, int]) -> int:
            r, c = dst
            return abs(target_row - r)

        best = min(move_actions, key=lambda x: score_move(x[1]))
        # Optionnel: parfois poser un mur
        if self.wall_prob > 0 and random.random() < self.wall_prob:
            wall_actions = [(a, w) for (a, w) in actions if a == "W"]
            if wall_actions:
                return random.choice([("W", w) for (_, w) in wall_actions])

        return ("M", best[1])
