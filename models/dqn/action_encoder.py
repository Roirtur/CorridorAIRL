from typing import Tuple, List
from corridor import Action

class ActionEncoder:
    """
    Maps Corridor actions to/from neural network action indices.
    
    Action Space Structure (for N=9 board):
    - Move actions: [0, N*N-1] = [0, 80]  (81 actions)
    - Horizontal walls: [N*N, N*N + (N-1)*(N-1) - 1] = [81, 144] (64 actions)
    - Vertical walls: [N*N + (N-1)*(N-1), N*N + 2*(N-1)*(N-1) - 1] = [145, 208] (64 actions)
    
    Total: 81 + 64 + 64 = 209 actions for 9x9 board
    """
    
    def __init__(self, board_size: int = 9):
        self.N = board_size
        
        # Calculate action space boundaries
        self.move_actions_start = 0
        self.move_actions_end = self.N * self.N
        
        self.h_wall_start = self.move_actions_end
        self.h_wall_end = self.h_wall_start + (self.N - 1) * (self.N - 1)
        
        self.v_wall_start = self.h_wall_end
        self.v_wall_end = self.v_wall_start + (self.N - 1) * (self.N - 1)
        
        self.action_space_size = self.v_wall_end
    
    def encode(self, action: Action) -> int:
        """
        Converts a Corridor action to a neural network index.
        """
        action_type = action[0]
        
        if action_type == "M":
            _, (r, c) = action
            return self.move_actions_start + r * self.N + c
        
        elif action_type == "W":
            _, (r, c, orientation) = action
            
            if orientation == "H":
                return self.h_wall_start + r * (self.N - 1) + c
            else:  # orientation == "V"
                return self.v_wall_start + r * (self.N - 1) + c
        
        else:
            raise ValueError(f"Unknown action type: {action_type}")
    
    def decode(self, action_idx: int) -> Action:
        """
        Converts a neural network output back to a Corridor action.
        """
        if action_idx < 0 or action_idx >= self.action_space_size:
            raise ValueError(f"Action index {action_idx} out of bounds [0, {self.action_space_size})")
        
        if action_idx < self.move_actions_end:
            # Move action
            idx = action_idx - self.move_actions_start
            r = idx // self.N
            c = idx % self.N
            return ("M", (r, c))
        
        elif action_idx < self.h_wall_end:
            # Horizontal wall
            idx = action_idx - self.h_wall_start
            r = idx // (self.N - 1)
            c = idx % (self.N - 1)
            return ("W", (r, c, "H"))
        
        else:
            # Vertical wall
            idx = action_idx - self.v_wall_start
            r = idx // (self.N - 1)
            c = idx % (self.N - 1)
            return ("W", (r, c, "V"))
    
    def encode_legal_actions(self, legal_actions: List[Action]) -> List[int]:
        """
        Encodes a list of legal actions to their indices.
        """
        return [self.encode(action) for action in legal_actions]
    
    def get_legal_action_mask(self, legal_actions: List[Action]) -> List[bool]:
        """
        Creates a boolean mask for legal actions.
        Useful for masking illegal actions in neural network output.
        """
        mask = [False] * self.action_space_size
        for action in legal_actions:
            idx = self.encode(action)
            mask[idx] = True
        return mask
    
    def filter_legal_actions(self, q_values: List[float], legal_actions: List[Action]) -> Tuple[int, Action]:
        """
        Selects the best legal action from Q-values.
        """
        if not legal_actions:
            raise ValueError("No legal actions available")
        
        legal_indices = self.encode_legal_actions(legal_actions)
        
        # Filter Q-values to only legal actions
        legal_q_values = [(q_values[idx], idx) for idx in legal_indices]
        
        # Select action with highest Q-value
        _, best_idx = max(legal_q_values, key=lambda x: x[0])
        
        best_action = self.decode(best_idx)
        
        return best_idx, best_action