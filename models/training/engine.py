import random
from typing import Any, Tuple, List, Union, Optional
from corridor import Corridor
from models.base_agent import BaseAgent
from models.utils.rewards import get_shaped_reward
from models.utils.persistence import save_model

def run_episode(env: Corridor, agent: Any, adversary: BaseAgent, training: bool = True) -> Tuple[int, int]:
    """
    Runs a single episode.
    Returns (win_status, steps).
    """
    obs = env.reset()
    
    # Randomize start player for training robustness
    my_id = random.choice([1, 2]) if training else 1
    
    # If we are P2, adversary moves first
    if my_id == 2:
        opp_action = adversary.select_action(env, obs)
        obs, _, done, info = env.step(opp_action)
        if done: return 0, env.move_count # Opponent won immediately (unlikely)

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
            return 1, env.move_count
        
        # 2. Opponent's turn
        opp_action = adversary.select_action(env, obs)
        obs, opp_reward, done, info = env.step(opp_action)
        
        if done:
            if training:
                agent.update(state, action, -1.0, None, None, True, env=None, next_legal_actions=None)
            return 0, env.move_count
        
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
        
    return 0, env.move_count

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
        win, _ = run_episode(env, agent, adversary, training=True)
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
