import os
import pickle
import random
from typing import Dict, Optional, Any, List, Union
from corridor import Corridor, Action
from .base_agent import BaseAgent

def get_canonical_state(obs: Dict, env: Corridor) -> tuple:
    """
    Converts observation to a canonical state tuple to reduce state space.
    Applies player perspective and horizontal symmetry.
    """
    N = obs['N']
    player = obs['to_play']
    
    # 1. Player perspective (always "me" vs "opponent")
    if player == 1:
        my_pos = obs['p1']
        opp_pos = obs['p2']
        my_walls = obs['walls_left'][1]
    else:
        # Flip board vertically for P2
        my_pos = (N - 1 - obs['p2'][0], obs['p2'][1])
        opp_pos = (N - 1 - obs['p1'][0], obs['p1'][1])
        my_walls = obs['walls_left'][2]
    
    # 2. Horizontal symmetry
    # We want to map states to the "left" side as much as possible.
    # If my_pos is on the right, flip.
    # If my_pos is in the center, check opp_pos.
    
    should_flip = False
    if my_pos[1] > N // 2:
        should_flip = True
    elif my_pos[1] == N // 2:
        if opp_pos[1] > N // 2:
            should_flip = True
            
    if should_flip:
        my_pos = (my_pos[0], N - 1 - my_pos[1])
        opp_pos = (opp_pos[0], N - 1 - opp_pos[1])
    
    # State: (my_pos, opp_pos, has_walls)
    # This is a compact representation that worked well for 5x5.
    # For 9x9, it might need more details (like distances), but let's start here.
    return (my_pos, opp_pos, my_walls > 0)

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
    data = {
        "q_table": dict(agent.q_table),
        "episodes": episodes,
        "board_size": env.N,
        "walls": env.walls_per_player,
        "adversary": adversary_name,
        "epsilon": agent.epsilon,
        "alpha": agent.alpha,
        "gamma": agent.gamma
    }
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
        
    agent.q_table.update(data.get("q_table", {}))
    agent.episodes_trained = data.get("episodes", 0)
    # We don't overwrite epsilon/alpha/gamma from file to allow tuning, 
    # unless we want to resume exactly. Let's keep current config but warn?
    # Actually, usually we want to load the Q-table.
    
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

    state = get_canonical_state(obs, env)
    action = agent.select_action(env, obs)
    
    phi_s = get_shaped_reward(env, my_id)
    
    done = False
    while not done:
        # 1. Execute our action
        obs, reward, done, info = env.step(action)
        
        if done:
            if training:
                agent.update(state, action, 1.0, None, None, True)
            return 1
        
        # 2. Opponent's turn
        opp_action = adversary.select_action(env, obs)
        obs, opp_reward, done, info = env.step(opp_action)
        
        if done:
            if training:
                agent.update(state, action, -1.0, None, None, True)
            return 0
        
        # 3. Prepare next step
        next_state = get_canonical_state(obs, env)
        next_action = agent.select_action(env, obs)
        
        if training:
            phi_next_s = get_shaped_reward(env, my_id)
            shaped_reward = 0.0 + agent.gamma * phi_next_s - phi_s
            agent.update(state, action, shaped_reward, next_state, next_action, False)
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
