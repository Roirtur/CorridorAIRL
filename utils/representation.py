from typing import Dict, Tuple
from corridor import Corridor 
import numpy as np

def bin_walls(walls_left: int) -> int:
    """
    Discrétisation des murs pour réduire la taille de la Q-Table.
    """
    if walls_left <= 2:
        return 0 # Critique
    elif walls_left <= 5:
        return 1 # Faible
    elif walls_left <= 8:
        return 2 # Moyen
    else:
        return 3 # Élevé

def calculate_adjacency_mask(obs: Dict, pos: Tuple[int, int]) -> int:
    """
    Encodage binaire des obstacles immédiats (Topologie locale).
    - Bit 0 (+1) : Nord bloqué
    - Bit 1 (+2) : Sud bloqué
    - Bit 2 (+4) : Ouest bloqué
    - Bit 3 (+8) : Est bloqué
    """
    r, c = pos
    N = obs["N"]
    mask = 0
    
    # North
    if r == 0 or (r - 1, c) in obs["H"]:
        mask |= 1 
    # South
    if r == N - 1 or (r, c) in obs["H"]:
        mask |= 2 
    # West
    if c == 0 or (r, c - 1) in obs["V"]:
        mask |= 4  
    # East
    if c == N - 1 or (r, c) in obs["V"]:
        mask |= 8 
        
    return mask

def flip_obs_for_symmetry(obs: Dict) -> Dict:
    """
    Applique une symétrie verticale au plateau pour normaliser l'état.
    Si le joueur actuel est 2 (monte vers le haut), on retourne le plateau pour qu'il "descende" vers le bas.
    Cela réduit l'espace d'états en traitant les situations symétriques comme identiques.
    """
    N = obs["N"]
    if obs["to_play"] == 1:
        return obs  # Pas de flip si joueur 1
    
    # Flip vertical : échanger p1 et p2, retourner positions, échanger H et V
    flipped = {
        "N": N,
        "to_play": 1,  # Après flip, on traite comme joueur 1
        "p1": (N - 1 - obs["p2"][0], obs["p2"][1]),  # opp devient p1
        "p2": (N - 1 - obs["p1"][0], obs["p1"][1]),  # me devient p2
        "walls_left": {1: obs["walls_left"][2], 2: obs["walls_left"][1]},  # Swap murs
        "H": {(N - 1 - r, c) for r, c in obs["V"]},  # V devient H après flip
        "V": {(N - 1 - r, c) for r, c in obs["H"]},  # H devient V après flip
        "move_count": obs["move_count"]
    }
    return flipped

def tabular_state_representation(env: "Corridor", obs: Dict) -> Tuple:
    """
    Construction de l'état pour le Q-Learning avec symétrie du plateau.
    Normalise l'état pour que le joueur actuel "descende" toujours vers le bas, réduisant les états symétriques.
    Combine : Position Normalisée + Vision Tactique (Mask) + Vision Stratégique (Distance) + Gestion Ressource (Murs)
    """
    
    # Appliquer la symétrie si nécessaire
    normalized_obs = flip_obs_for_symmetry(obs)
    
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