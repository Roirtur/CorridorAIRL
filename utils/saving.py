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
    
def generate_path_name():
    pass