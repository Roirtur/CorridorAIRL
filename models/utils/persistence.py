import os
import pickle
from typing import Any
from corridor import Corridor

def generate_save_path(base_dir: str, model_name: str, board_size: int, episodes: int, adversary_name: str) -> str:
    """Generates a structured filename for the model."""
    filename = f"{model_name}_N{board_size}_E{episodes}_vs_{adversary_name}.pkl"
    return os.path.join(base_dir, filename)

def save_model(agent: Any, path: str, env: Corridor, adversary_name: str, episodes: int):
    """Saves the agent model and metadata."""
    # Import DQNAgent locally to avoid circular imports if possible, 
    # or check class name string to be safe.
    
    data = {
        "episodes": episodes,
        "board_size": env.N,
        "walls": env.walls_per_player,
        "adversary": adversary_name,
        "epsilon": agent.epsilon,
        "alpha": getattr(agent, 'alpha', None),
        "gamma": getattr(agent, 'gamma', None)
    }
    
    # Handle DQN agent differently (save neural network weights)
    # We check class name to avoid importing DQNAgent which might cause circular dependency
    if agent.__class__.__name__ in ["DQNAgent", "MyAgent"]:
        data["agent_type"] = "DQN"
        data["q_network_state"] = agent.q_network.state_dict()
        data["target_network_state"] = agent.target_network.state_dict()
        data["optimizer_state"] = agent.optimizer.state_dict()
    else:
        # Q-learning agents (SARSA, etc.)
        data["agent_type"] = "Q-Learning"
        data["q_table"] = dict(agent.q_table)
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"Model saved to {path}")

def load_model(agent: Any, path: str):
    """Loads the agent model and metadata."""
    
    if not os.path.exists(path):
        print(f"Warning: Model file {path} not found. Starting with empty model.")
        return
        
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    agent_type = data.get("agent_type", "Q-Learning")
    
    if agent_type == "DQN" and agent.__class__.__name__ in ["DQNAgent", "MyAgent"]:
        agent.q_network.load_state_dict(data["q_network_state"])
        agent.target_network.load_state_dict(data["target_network_state"])
        agent.optimizer.load_state_dict(data["optimizer_state"])
    elif agent_type == "Q-Learning":
        agent.q_table.update(data.get("q_table", {}))
    
    agent.episodes_trained = data.get("episodes", 0)
    
    print(f"Model loaded from {path}")
    print(f"  Episodes: {data.get('episodes')}")
    print(f"  Board Size: {data.get('board_size')}")
    print(f"  Adversary: {data.get('adversary')}")
