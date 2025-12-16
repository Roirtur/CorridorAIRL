from typing import Dict, Tuple
import numpy as np
from corridor import Corridor, Action

def get_representation_state(obs: Dict, env: Corridor) -> tuple:
    """
    Optimized compact state representation for Tabular RL.
    Reduces state space by using path lengths instead of wall configurations
    and exploiting horizontal symmetry.
    
    State Tuple:
    (my_r, my_c, opp_r, opp_c, my_walls_bucket, opp_walls_bucket, my_path_len, opp_path_len)
    """
    N = obs['N']
    player = obs['to_play']
    
    # 1. Vertical Perspective Alignment (P2 -> P1 view)
    if player == 1:
        my_pos = obs['p1']
        opp_pos = obs['p2']
        my_walls = obs['walls_left'][1]
        opp_walls = obs['walls_left'][2]
        my_id, opp_id = 1, 2
    else:
        # Flip board vertically for P2
        my_pos = (N - 1 - obs['p2'][0], obs['p2'][1])
        opp_pos = (N - 1 - obs['p1'][0], obs['p1'][1])
        my_walls = obs['walls_left'][2]
        opp_walls = obs['walls_left'][1]
        my_id, opp_id = 2, 1

    # 2. Horizontal Symmetry (Force 'my_pos' to left side)
    # If my column is > N//2, flip horizontally
    if my_pos[1] > N // 2:
        my_pos = (my_pos[0], N - 1 - my_pos[1])
        opp_pos = (opp_pos[0], N - 1 - opp_pos[1])
        # Note: We don't need to flip walls because we only track counts/buckets

    # 3. Path Features (Crucial for navigation without wall map)
    # We use the env to calculate path lengths.
    my_path = env.shortest_path_length(my_id)
    opp_path = env.shortest_path_length(opp_id)
    
    # 4. Wall Buckets (0 or >0)
    # Exact count matters less than "can I place a wall?"
    my_walls_bucket = 1 if my_walls > 0 else 0
    opp_walls_bucket = 1 if opp_walls > 0 else 0

    return (
        my_pos[0], my_pos[1],
        opp_pos[0], opp_pos[1],
        my_walls_bucket,
        opp_walls_bucket,
        my_path,
        opp_path
    )

def state_to_tensor(state: tuple, board_size: int) -> np.ndarray:
    """
    Converts the unified state tuple to a normalized numpy array for Neural Networks.
    """
    N = float(board_size)
    # Normalize positions by N
    # Wall buckets are 0/1
    # Path lengths normalized by N*N (approx max path)
    
    return np.array([
        state[0] / N, state[1] / N,       # My Pos
        state[2] / N, state[3] / N,       # Opp Pos
        float(state[4]), float(state[5]), # Wall Buckets
        state[6] / (N*N), state[7] / (N*N) # Path lengths
    ], dtype=np.float32)

def get_grid_state(obs: Dict, env: Corridor) -> tuple:
    """
    Returns (state_tensor, is_flipped)
    state_tensor is a (4, N, N) numpy array.
    Channels:
    0: My Position (1.0 at pos)
    1: Opponent Position (1.0 at pos)
    2: Horizontal Walls (1.0 where wall exists)
    3: Vertical Walls (1.0 where wall exists)
    """
    N = env.N
    player = obs['to_play']
    is_flipped = (player == 2)
    
    # Grids
    my_pos_grid = np.zeros((N, N), dtype=np.float32)
    opp_pos_grid = np.zeros((N, N), dtype=np.float32)
    h_walls_grid = np.zeros((N, N), dtype=np.float32)
    v_walls_grid = np.zeros((N, N), dtype=np.float32)
    
    p1 = obs['p1']
    p2 = obs['p2']
    H = env.H
    V = env.V
    
    if not is_flipped:
        my_pos_grid[p1] = 1.0
        opp_pos_grid[p2] = 1.0
        for (r, c) in H:
            h_walls_grid[r, c] = 1.0
        for (r, c) in V:
            v_walls_grid[r, c] = 1.0
    else:
        # Flip perspective
        my_r, my_c = p2
        my_pos_grid[N - 1 - my_r, my_c] = 1.0
        
        opp_r, opp_c = p1
        opp_pos_grid[N - 1 - opp_r, opp_c] = 1.0
        
        for (r, c) in H:
            if 0 <= N - r - 2 < N:
                h_walls_grid[N - r - 2, c] = 1.0
        
        for (r, c) in V:
            v_walls_grid[N - 1 - r, c] = 1.0
            
    state = np.stack([my_pos_grid, opp_pos_grid, h_walls_grid, v_walls_grid])
    return (state, is_flipped)

def flip_action(action: Action, N: int) -> Action:
    """
    Flips an action from P2's perspective to P1's canonical perspective.
    """
    kind = action[0]
    if kind == "M":
        _, (r, c) = action
        return ("M", (N - 1 - r, c))
    elif kind == "W":
        _, (r, c, ori) = action
        if ori == "H":
            # H wall at r blocks r, r+1.
            # Flipped blocks N-1-r, N-r-2.
            # Wall at k blocks k, k+1. So k = N-r-2.
            return ("W", (N - r - 2, c, "H"))
        else:
            # V wall at r blocks r, r (col c, c+1).
            # Flipped blocks N-1-r.
            return ("W", (N - 1 - r, c, "V"))
    return action

def action_to_features(action: Action, board_size: int) -> np.ndarray:
    """
    Convert an action to a simple feature vector.
    [is_move, is_wall, r_norm, c_norm, is_horizontal, is_vertical]
    """
    N = float(board_size)
    features = []
    
    kind = action[0]
    if kind == "M":
        _, (r, c) = action
        features = [1.0, 0.0, r/N, c/N, 0.0, 0.0]
    elif kind == "W":
        _, (r, c, ori) = action
        is_h = 1.0 if ori == "H" else 0.0
        is_v = 1.0 if ori == "V" else 0.0
        features = [0.0, 1.0, r/N, c/N, is_h, is_v]
    
    return np.array(features, dtype=np.float32)
