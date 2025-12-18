from typing import Dict, Tuple
from corridor import Corridor 
import numpy as np


def tabular_state_representation(env: Corridor, obs: Dict) -> Tuple:
    
    p1_pos = obs["p1"]
    p2_pos = obs["p2"]

    # On utilise la même fonction logique pour analyser l'environnement des deux joueurs.
    # Cela permet à l'agent de voir ses propres blocages ET ceux de l'adversaire.
    mask_p1 = calculate_adjacency_mask(obs, p1_pos)
    mask_p2 = calculate_adjacency_mask(obs, p2_pos)

    # Nouvelles features : distances de chemin les plus courtes
    dist_p1 = env.shortest_path_length(1)  # Distance pour P1 vers sa ligne but
    dist_p2 = env.shortest_path_length(2)  # Distance pour P2 vers sa ligne but

    # Features de proximité relative
    manhattan_dist = abs(p1_pos[0] - p2_pos[0]) + abs(p1_pos[1] - p2_pos[1])
    # same_row = 1 if p1_pos[0] == p2_pos[0] else 0
    # same_col = 1 if p1_pos[1] == p2_pos[1] else 0
    # total_walls_placed = (env.walls_per_player - obs["walls_left"][1]) + (env.walls_per_player - obs["walls_left"][2])

    walls_left = int(obs["walls_left"] == 0)

    # 3. Construction du Tuple final (ajout des distances et proximités)
    return (
        obs["to_play"],       # Qui a la main ?
        p1_pos,               # Où est P1 ?
        p2_pos,               # Où est P2 ?
        #obs["walls_left"][1], # Stock P1
        obs["walls_left"][2], # Stock P2
        dist_p1,              # Distance chemin P1
        dist_p2,              # Distance chemin P2
        manhattan_dist,       # Distance Manhattan entre joueurs
        # same_row,             # Même ligne ?
        # same_col,             # Même colonne ?
        # total_walls_placed,   # Murs placés au total
        walls_left,
        mask_p1,              # Obstacles immédiats autour de P1
        mask_p2               # Obstacles immédiats autour de P2
    )


def calculate_adjacency_mask(obs: Dict, pos: Tuple[int, int]) -> int:
    """
    Binary encoding for player obstacle :
    - Bit 0 (+1) : N
    - Bit 1 (+2) : S
    - Bit 2 (+4) : W
    - Bit 3 (+8) : E
    """
    r, c = pos
    
    mask = 0
    
    #north
    if r == 0 or (r - 1, c) in obs["H"]:
        mask |= 1 
        
    #south
    if r == obs["N"] - 1 or (r, c) in obs["H"]:
        mask |= 2 
        
    #west
    if c == 0 or (r, c - 1) in obs["V"]:
        mask |= 4  

    #east
    if c == obs["N"] - 1 or (r, c) in obs["V"]:
        mask |= 8 
        
    return mask

def approximation_agent_state_representation(obs: Dict, player: int = 1) -> np.ndarray:
    """
    Returns a stack of binary feature planes representing:
    - Player 1 position
    - Player 2 position  
    - Horizontal walls
    - Vertical walls
    - Walls remaining (player)
    - Walls remaining (opponent)
    
    Shape: (N, N, 6) flattened to (N*N*6,)
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
    
    # Stack feature planes (channels-last format)
    state = np.stack([
        p1_plane, 
        p2_plane, 
        h_walls_plane, 
        v_walls_plane, 
        walls_remaining_player,
        walls_remaining_opponent
    ], axis=-1)
    
    return state.flatten()