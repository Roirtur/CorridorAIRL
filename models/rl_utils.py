import os
import pickle
import random
from typing import Dict, Optional, Any, List, Union
from corridor import Corridor, Action
from .base_agent import BaseAgent
import numpy as np

def get_representation_state(obs: Dict, env: Corridor) -> tuple:
    """
    Unified state representation for all agents.
    Returns a tuple: (my_r, my_c, opp_r, opp_c, my_walls, opp_walls, valid_n, valid_s, valid_w, valid_e)
    
    This representation is:
    1. Perspective-aligned (P2 is flipped to look like P1)
    2. Simple (no horizontal symmetry)
    3. Neural-network friendly (can be directly converted to tensor)
    """
    N = obs['N']
    player = obs['to_play']
    
    # 1. Player perspective (always "me" vs "opponent")
    # We align everything so "me" starts at top (0, N//2) and goes to bottom (N-1, *)
    if player == 1:
        my_pos = obs['p1']
        opp_pos = obs['p2']
        my_walls = obs['walls_left'][1]
        opp_walls = obs['walls_left'][2]
    else:
        # Flip board vertically for P2
        my_pos = (N - 1 - obs['p2'][0], obs['p2'][1])
        opp_pos = (N - 1 - obs['p1'][0], obs['p1'][1])
        my_walls = obs['walls_left'][2]
        opp_walls = obs['walls_left'][1]

    # 2. Valid moves (local view)
    # We need to check valid moves in the *canonical* perspective
    real_p = obs['p1'] if player == 1 else obs['p2']
    r, c = real_p
    
    # Real directions: N(-1,0), S(1,0), W(0,-1), E(0,1)
    # env._can_step checks if the move is valid (bounds + walls)
    # We use 1.0 for valid, 0.0 for invalid to be NN friendly
    can_n = 1.0 if env._can_step(real_p, (r-1, c)) else 0.0
    can_s = 1.0 if env._can_step(real_p, (r+1, c)) else 0.0
    can_w = 1.0 if env._can_step(real_p, (r, c-1)) else 0.0
    can_e = 1.0 if env._can_step(real_p, (r, c+1)) else 0.0
    
    # Transform to canonical perspective if we flipped for P2
    if player == 2:
        # Vertical flip: North <-> South
        can_n, can_s = can_s, can_n
        
    return (
        float(my_pos[0]), float(my_pos[1]), 
        float(opp_pos[0]), float(opp_pos[1]), 
        float(my_walls), float(opp_walls), 
        can_n, can_s, can_w, can_e
    )

def state_to_tensor(state: tuple, board_size: int) -> np.ndarray:
    """
    Converts the unified state tuple to a normalized numpy array for Neural Networks.
    """
    N = float(board_size)
    # Normalize positions by N
    # Normalize walls by 10 (max walls)
    # Valid moves are already 0/1
    
    return np.array([
        state[0] / N, state[1] / N,       # My Pos
        state[2] / N, state[3] / N,       # Opp Pos
        state[4] / 10.0, state[5] / 10.0, # Walls
        state[6], state[7], state[8], state[9] # Valid moves
    ], dtype=np.float32)

def get_shaped_reward(env: Corridor, my_id: int) -> float:
    """
    Calculates potential-based reward shaping.
    Phi(s) = 0.2 * OppDist - 1.0 * MyDist
    """
    opp_id = 2 if my_id == 1 else 1
    my_d = env.shortest_path_length(my_id)
    opp_d = env.shortest_path_length(opp_id)
    return 0.2 * opp_d - 1.0 * my_d

def generate_save_path(base_dir: str, model_name: str, board_size: int, episodes: int, adversary_name: str) -> str:
    """Generates a structured filename for the model."""
    filename = f"{model_name}_N{board_size}_E{episodes}_vs_{adversary_name}.pkl"
    return os.path.join(base_dir, filename)

def save_model(agent: Any, path: str, env: Corridor, adversary_name: str, episodes: int):
    """Saves the agent model and metadata."""
    from models.dqn_agent import DQNAgent
    import torch
    
    data = {
        "episodes": episodes,
        "board_size": env.N,
        "walls": env.walls_per_player,
        "adversary": adversary_name,
        "epsilon": agent.epsilon,
        "alpha": agent.alpha,
        "gamma": agent.gamma
    }
    
    # Handle DQN agent differently (save neural network weights)
    if isinstance(agent, DQNAgent):
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
    from models.dqn_agent import DQNAgent
    import torch
    
    if not os.path.exists(path):
        print(f"Warning: Model file {path} not found. Starting with empty model.")
        return
        
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    agent_type = data.get("agent_type", "Q-Learning")
    
    if agent_type == "DQN" and isinstance(agent, DQNAgent):
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

def run_episode(env: Corridor, agent: Any, adversary: BaseAgent, training: bool = True) -> int:
    """
    Runs a single episode.
    Returns 1 if agent wins, 0 otherwise.
    """
    obs = env.reset()
    
    # Randomize start player for training robustness
    my_id = random.choice([1, 2]) if training else 1
    
    # If we are P2, adversary moves first
    if my_id == 2:
        opp_action = adversary.select_action(env, obs)
        obs, _, done, info = env.step(opp_action)
        if done: return 0 # Opponent won immediately (unlikely)

    # Unified state for everyone
    state = agent.preprocess_state(env, obs)
    action = agent.select_action(env, obs)
    
    phi_s = get_shaped_reward(env, my_id)
    
    done = False
    while not done:
        # 1. Execute our action
        obs, reward, done, info = env.step(action)
        
        if done:
            if training:
                agent.update(state, action, 1.0, None, None, True, env=None, next_legal_actions=None)
            return 1
        
        # 2. Opponent's turn
        opp_action = adversary.select_action(env, obs)
        obs, opp_reward, done, info = env.step(opp_action)
        
        if done:
            if training:
                agent.update(state, action, -1.0, None, None, True, env=None, next_legal_actions=None)
            return 0
        
        # 3. Prepare next step
        next_state = agent.preprocess_state(env, obs)
        next_legal_actions = env.legal_actions()
        next_action = agent.select_action(env, obs)
        
        if training:
            phi_next_s = get_shaped_reward(env, my_id)
            shaped_reward = 0.0 + agent.gamma * phi_next_s - phi_s
            
            agent.update(state, action, shaped_reward, next_state, next_action, False, env=env, next_legal_actions=next_legal_actions)
            phi_s = phi_next_s
        
        state = next_state
        action = next_action
        
    return 0

def train_loop(agent: Any, env: Corridor, adversaries: Union[List[BaseAgent], BaseAgent], episodes: int, save_path: Optional[str] = None, start_epsilon: float = 1.0, end_epsilon: float = 0.1, alpha: float = 0.1, gamma: float = 0.995, epsilon_decay: Optional[float] = None):
    """
    Generic training loop.
    """
    if not isinstance(adversaries, list):
        adversaries = [adversaries]
        
    adversary_names = ", ".join([adv.name for adv in adversaries])
    print(f"Starting training for {episodes} episodes against {adversary_names}...")
    
    # Reset/Set hyperparameters
    agent.epsilon = start_epsilon
    if hasattr(agent, 'alpha'):
        agent.alpha = alpha
    if hasattr(agent, 'gamma'):
        agent.gamma = gamma
        
    # Calculate linear decay step if no multiplicative decay is provided
    linear_decay_step = 0
    if epsilon_decay is None:
        linear_decay_step = (start_epsilon - end_epsilon) / episodes
    
    wins = 0
    
    for episode in range(1, episodes + 1):
        adversary = random.choice(adversaries)
        win = run_episode(env, agent, adversary, training=True)
        wins += win
        
        # Decay epsilon
        if epsilon_decay is not None:
            # Multiplicative decay
            agent.epsilon = max(end_epsilon, agent.epsilon * epsilon_decay)
        elif agent.epsilon > end_epsilon:
            # Linear decay
            agent.epsilon -= linear_decay_step
            
        agent.episodes_trained += 1
        
        if episode % 100 == 0:
            print(f"Episode {episode}/{episodes} - Win Rate (last 100): {wins}% - Epsilon: {agent.epsilon:.3f}")
            wins = 0
            
    if save_path:
        save_model(agent, save_path, env, adversary_names, agent.episodes_trained)
