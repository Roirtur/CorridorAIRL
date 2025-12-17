from typing import Dict


def tabular_state_representation(obs: Dict):
    
    state = (
        obs["p1"],
        obs["p2"],
        obs["walls_left"][1],
        obs["walls_left"][2],
        # Ignores wall positions for now to simplify
        # obs["H"],
        # obs["V"],
    )

    return state