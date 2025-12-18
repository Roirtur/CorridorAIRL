from typing import Dict
import numpy as np


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

def approximation_agent_state_representation(obs: Dict, player: int = 1) -> np.ndarray:
    """
    Returns a stack of binary feature planes representing:
    - Player 1 position
    - Player 2 position  
    - Horizontal walls
    - Vertical walls
    - Walls remaining (player)
    - Walls remaining (opponent)
    
    Shape: (6, N, N) in PyTorch channels first format
    """    
    N = obs["N"]
    
    # feature planes
    p1_plane = np.zeros((N, N), dtype=np.float32)
    p2_plane = np.zeros((N, N), dtype=np.float32)
    h_walls_plane = np.zeros((N, N), dtype=np.float32)
    v_walls_plane = np.zeros((N, N), dtype=np.float32)
    walls_remaining_player = np.zeros((N, N), dtype=np.float32)
    walls_remaining_opponent = np.zeros((N, N), dtype=np.float32)
    
    # pawn positions
    p1_plane[obs["p1"]] = 1.0
    p2_plane[obs["p2"]] = 1.0
    
    # horizontal walls (mark both cells blocked by the wall)
    for (r, c) in obs["H"]:
        if r < N and c < N:
            h_walls_plane[r, c] = 1.0
        if r < N and c + 1 < N:
            h_walls_plane[r, c + 1] = 1.0
    
    # vertical walls (mark both cells blocked by the wall)
    for (r, c) in obs["V"]:
        if r < N and c < N:
            v_walls_plane[r, c] = 1.0
        if r + 1 < N and c < N:
            v_walls_plane[r + 1, c] = 1.0
    
    # walls remaining (normalized by max walls: 10)
    opponent = 2 if player == 1 else 1
    walls_remaining_player.fill(obs["walls_left"][player] / 10.0)
    walls_remaining_opponent.fill(obs["walls_left"][opponent] / 10.0)
    
    # stack feature planes in PyTorch format
    state = np.stack([
        p1_plane, 
        p2_plane, 
        h_walls_plane, 
        v_walls_plane, 
        walls_remaining_player,
        walls_remaining_opponent
    ], axis=0)
    
    return state