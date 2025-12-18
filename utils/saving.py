import pickle
import json
import os
from typing import Dict, List, Any

def save_tabular_model(agent, path: str):
    """
    Saves the Q-table of a tabular agent to a pickle file
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, 'wb') as f:
            pickle.dump(dict(agent.q_table), f)
        print(f"Model saved to {path}")
    except AttributeError:
        print(f"Agent {agent.name} does not have a q_table attribute to save.")

def load_tabular_model(agent, path: str):
    """
    Loads the Q-table into a tabular agent from a pickle file
    """
    if not os.path.exists(path):
        print(f"No model found at {path}, starting with empty Q-table.")
        return
    
    try:
        with open(path, 'rb') as f:
            q_table_dict = pickle.load(f)
            
            agent.q_table.update(q_table_dict)

    except AttributeError:
        print(f"Agent {agent.name} does not have a q_table attribute.")
    except Exception as e:
        print(f"Error loading model from {path}: {e}")

def save_training_data(data: Dict[str, List[float]], path: str):
    """
    Saves training metrics to a JSON file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Training data saved to {path}")

def load_training_data(path: str) -> Dict[str, List[float]]:
    """
    Loads training metrics from a JSON file.
    """
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return json.load(f)
    
def generate_path_name(agent_name: str, total_episodes: int, opponent_name: str, file_type: str, board_size: int = None) -> str:
    """
    Generates a filename with format {model name}_B{board_size}_E{number episode}_VS{opponent}.{pkl|json}
    If board_size is provided, it's included in the filename.
    """
    ext = "pkl" if file_type == "model" else "json"
    folder = "saved_models"
    
    # Ensure directory exists
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Include board size if provided
    if board_size:
        filename = f"{agent_name}_B{board_size}_E{total_episodes}_VS{opponent_name}.{ext}"
    else:
        filename = f"{agent_name}_E{total_episodes}_VS{opponent_name}.{ext}"
        
    return os.path.join(folder, filename)

def parse_model_info(filename: str) -> dict:
    """
    Parses model filename to extract agent type, board size, episodes, and opponent.
    Expected format: {agent_name}_B{board_size}_E{episodes}_VS{opponent}.pkl
    or: {agent_name}_E{episodes}_VS{opponent}.pkl (legacy format without board size)
    """
    import re
    
    # Remove extension
    name = os.path.splitext(os.path.basename(filename))[0]
    
    # Try to parse with board size
    pattern_with_board = r'^(.+?)_B(\d+)_E(\d+)_VS(.+)$'
    match = re.match(pattern_with_board, name)
    
    if match:
        return {
            'agent_name': match.group(1),
            'board_size': int(match.group(2)),
            'episodes': int(match.group(3)),
            'opponent': match.group(4)
        }
    
    # Try legacy format without board size
    pattern_legacy = r'^(.+?)_E(\d+)_VS(.+)$'
    match = re.match(pattern_legacy, name)
    
    if match:
        return {
            'agent_name': match.group(1),
            'board_size': None,
            'episodes': int(match.group(2)),
            'opponent': match.group(3)
        }
    
    # Could not parse, return minimal info
    return {
        'agent_name': name,
        'board_size': None,
        'episodes': None,
        'opponent': None
    }
