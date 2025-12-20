# Made with AI

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


def save_approximation_agent_model(agent, path: str):
    """
    Saves the NN weights of an agent (e.g., DQN) to a .pth file.
    Stores q_network, target_network, optimizer state, and hyperparameters.
    """
    import torch
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Ensure .pth extension for PyTorch models
    if not path.endswith('.pth'):
        path = path.replace('.pkl', '.pth') if path.endswith('.pkl') else path + '.pth'
    
    try:
        # Build checkpoint with all necessary components
        checkpoint = {
            'q_network_state_dict': agent.q_network.state_dict(),
            'target_network_state_dict': agent.target_network.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'hyperparameters': {
                'board_size': agent.board_size,
                'gamma': agent.gamma,
                'epsilon': agent.epsilon,
                'epsilon_decay': agent.epsilon_decay,
                'min_epsilon': agent.min_epsilon,
                'batch_size': agent.batch_size,
                'target_update_freq': agent.target_update_freq,
                'learn_step': agent.learn_step,
            },
            'agent_type': type(agent).__name__,
            'agent_name': agent.name,
        }
        
        torch.save(checkpoint, path)
        print(f"Approximation model saved to {path}")
        
    except AttributeError as e:
        print(f"Agent {agent.name} does not have required neural network attributes: {e}")
    except Exception as e:
        print(f"Error saving approximation model to {path}: {e}")


def load_approximation_agent_model(agent, path: str):
    """
    Loads the NN weights into an agent (e.g., DQN) from a .pth file.
    Restores q_network, target_network, optimizer state, and hyperparameters.
    """
    import torch
    
    # Handle .pkl to .pth conversion for backward compatibility
    if path.endswith('.pkl') and not os.path.exists(path):
        alt_path = path.replace('.pkl', '.pth')
        if os.path.exists(alt_path):
            path = alt_path
    elif not path.endswith('.pth'):
        path = path + '.pth'
    
    if not os.path.exists(path):
        print(f"No model found at {path}, starting with randomly initialized weights.")
        return
    
    try:
        # Load checkpoint
        checkpoint = torch.load(path, map_location=agent.device)
        
        # Restore network weights
        agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        agent.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore hyperparameters if available
        if 'hyperparameters' in checkpoint:
            hyperparams = checkpoint['hyperparameters']
            agent.epsilon = hyperparams.get('epsilon', agent.epsilon)
            agent.learn_step = hyperparams.get('learn_step', 0)
            
        # Set networks to appropriate mode
        if agent.training_mode:
            agent.q_network.train()
        else:
            agent.q_network.eval()
            agent.epsilon = 0.0  # No exploration in evaluation mode
            
        agent.target_network.eval()
        
        print(f"Approximation model loaded from {path}")
        
    except AttributeError as e:
        print(f"Agent {agent.name} does not have required neural network attributes: {e}")
    except Exception as e:
        print(f"Error loading approximation model from {path}: {e}")
