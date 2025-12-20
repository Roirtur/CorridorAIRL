from typing import Dict, Tuple
from corridor import Corridor 
import numpy as np

def bin_walls(walls_left: int) -> int:
    """
    Walls discretization, use intervals rather than the exact number left
    """
    if walls_left <= 2:
        return 0 
    elif walls_left <= 5:
        return 1 
    elif walls_left <= 8:
        return 2 
    else:
        return 3 

def calculate_adjacency_mask(obs: Dict, pos: Tuple[int, int]) -> int:
    """
    Binary encoding for immediate wall obstacle.
    """
    r, c = pos
    mask = 0

    # North
    if r == 0 or (r - 1, c) in obs["H"]:
        mask |= 1 
    # South
    if r == obs["N"] - 1 or (r, c) in obs["H"]:
        mask |= 2 
    # West
    if c == 0 or (r, c - 1) in obs["V"]:
        mask |= 4  
    # East
    if c == obs["N"] - 1 or (r, c) in obs["V"]:
        mask |= 8 
        
    return mask

def vertical_symmetry_obs_flip(obs: Dict) -> Dict:
    """
    Vertical symmetry to normalize the board state
    """
    N = obs["N"]
    if obs["to_play"] == 1: #no flip for player 1
        return obs 
    
    flipped_obs = {
        "N": N,
        "to_play": 1,  
        "p1": (N - 1 - obs["p2"][0], obs["p2"][1]), 
        "p2": (N - 1 - obs["p1"][0], obs["p1"][1]),  
        "walls_left": {1: obs["walls_left"][2], 2: obs["walls_left"][1]}, 
        "H": {(N - 1 - r, c) for r, c in obs["V"]},  
        "V": {(N - 1 - r, c) for r, c in obs["H"]}, 
        "move_count": obs["move_count"]
    }
    return flipped_obs

def tabular_state_representation(env: "Corridor", obs: Dict) -> Tuple:
    """
    Construction de l'état pour le Q-Learning avec symétrie du plateau.
    Normalise l'état pour que le joueur actuel "descende" toujours vers le bas, réduisant les états symétriques.
    Combine : Position Normalisée + Vision Tactique (Mask) + Vision Stratégique (Distance) + Gestion Ressource (Murs)
    """
    
    # Appliquer la symétrie si nécessaire
    normalized_obs =  vertical_symmetry_obs_flip(obs)
    
    # 1. Identification (après normalisation, me est toujours 1)
    me = normalized_obs["to_play"]  # Toujours 1 après flip
    opp = 3 - me
    
    me_pos = normalized_obs[f"p{me}"]
    opp_pos = normalized_obs[f"p{opp}"]
    
    # 2. Vision Tactique (Masques immédiats avec obs normalisé)
    mask_me = calculate_adjacency_mask(normalized_obs, me_pos)
    mask_opp = calculate_adjacency_mask(normalized_obs, opp_pos)
    
    # 3. Vision Stratégique (Distances réelles via BFS du moteur sur env original, car env n'est pas modifié)
    # Note : Utiliser env original pour distances, car flip n'affecte pas les calculs de chemin
    d_me = env.shortest_path_length(obs["to_play"])  # Utiliser to_play original
    d_opp = env.shortest_path_length(3 - obs["to_play"])
    
    dist_me = d_me if d_me is not None else 99
    dist_opp = d_opp if d_opp is not None else 99
    
    # 4. Gestion des ressources (Binning sur obs original)
    walls_me_binned = bin_walls(obs["walls_left"][obs["to_play"]])  # to_play original
    
    # 5. Construction du Tuple symétrique
    return (
        me_pos,               # Position de me (normalisée)
        opp_pos,              # Position de opp (normalisée)
        walls_me_binned,      # Mes ressources (binned)
        dist_me,              # Ma distance au but
        dist_opp,             # Sa distance au but
        mask_me,              # Mes blocages locaux
        mask_opp              # Ses blocages locaux
    )

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