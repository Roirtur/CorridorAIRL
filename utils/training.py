import random
import math
import numpy as np
from typing import List, Tuple, Union
from corridor import Corridor
from models import BaseAgent
from utils.saving import save_training_data

def run_eval_game(env: Corridor, agent: BaseAgent, opponent: BaseAgent) -> float:
    """
    Runs one evaluation game. Returns 1.0 if agent wins, 0.0 otherwise.
    """
    obs = env.reset()
    done = False
    
    # Randomize sides
    agent_is_p1 = random.random() < 0.5
    p1 = agent if agent_is_p1 else opponent
    p2 = opponent if agent_is_p1 else agent
    
    steps = 0
    while not done and steps < 500: 
        player = obs["to_play"]
        current_agent = p1 if player == 1 else p2
        
        action = current_agent.select_action(env, obs)
        obs, _, done, info = env.step(action)
        steps += 1
        
    winner = info.get("winner")
    if winner == 1:
        return 1.0 if agent_is_p1 else 0.0
    elif winner == 2:
        return 1.0 if not agent_is_p1 else 0.0
    return 0.0 # Draw

def evaluate_agent(
    env: Corridor, 
    agent: BaseAgent, 
    opponents: List[BaseAgent], 
    n_games_per_opponent: int = 10
) -> float:
    """
    Evaluates agent against a list of opponents.
    Returns average win rate across all games.
    """
    old_epsilon = getattr(agent, 'epsilon', 0.0)
    old_training_mode = getattr(agent, 'training_mode', True)
    
    if hasattr(agent, 'epsilon'): agent.epsilon = 0.0
    if hasattr(agent, 'training_mode'): agent.training_mode = False
    if hasattr(agent, 'q_network'): agent.q_network.eval()

    total_wins = 0
    total_games = 0
    
    for opp in opponents:
        for _ in range(n_games_per_opponent):
            total_wins += run_eval_game(env, agent, opp)
            total_games += 1
            
    if hasattr(agent, 'epsilon'): agent.epsilon = old_epsilon
    if hasattr(agent, 'training_mode'): agent.training_mode = old_training_mode
    if hasattr(agent, 'q_network'): agent.q_network.train()
    
    return total_wins / total_games if total_games > 0 else 0.0

def training_loop(
    env: Corridor,
    agent: BaseAgent,
    opponents_schedule: List[Tuple[Union[BaseAgent, List[BaseAgent]], int]],
    save_path_model: str,
    save_path_data: str,
    eval_interval: int = 250,
    save_interval: int = 1000,
    min_epsilon: float = 0.05,
    initial_epsilon: float = 1.0
):
    """
    Advanced Training Loop.
    
    Args:
        opponents_schedule: List of tuples (Opponent(s), n_episodes). 
                            Opponent can be a single agent or a list of agents (mixed play).
    """
    
    total_episodes_done = 0
    
    total_planned_episodes = sum(n for _, n in opponents_schedule)
    
    history = {
        "rewards": [],
        "lengths": [],
        "win_rates": [],
        "eval_episodes": [],
        "epsilon": []
    }
    
    print(f"Starting training for {agent.name}...")
    print(f"Total planned episodes: {total_planned_episodes}")
    
    if hasattr(agent, "epsilon"):
        agent.epsilon = initial_epsilon

    phase_idx = 1
    
    for opponents_source, n_episodes in opponents_schedule:
        if isinstance(opponents_source, list):
            current_opponents = opponents_source
            phase_name = "Mixed Pool (" + ", ".join(o.name for o in current_opponents) + ")"
        else:
            current_opponents = [opponents_source]
            phase_name = opponents_source.name

        print(f"\n=== Phase {phase_idx}: vs {phase_name} ({n_episodes} episodes) ===")
        
        recent_rewards = []
        
        for i in range(n_episodes):
            current_episode_global = total_episodes_done + i + 1
            
            if hasattr(agent, "epsilon"):
                progress = current_episode_global / total_planned_episodes
                if progress < 0.9:
                    agent.epsilon = initial_epsilon - (initial_epsilon - min_epsilon) * (progress / 0.9)
                else:
                    agent.epsilon = min_epsilon
            
            opponent = random.choice(current_opponents)
            
            agent_player = 1 if random.random() < 0.5 else 2
            stats = agent.run_episode(env, opponent, agent_player=agent_player)

            history["rewards"].append(stats["reward"])
            history["lengths"].append(stats["steps"])
            recent_rewards.append(stats["reward"])
            
            if current_episode_global % eval_interval == 0:
                win_rate = evaluate_agent(env, agent, current_opponents)
                
                avg_reward = sum(recent_rewards[-eval_interval:]) / eval_interval
                eps_val = getattr(agent, 'epsilon', 0.0)
                
                history["win_rates"].append(win_rate)
                history["eval_episodes"].append(current_episode_global)
                history["epsilon"].append(eps_val)
                
                print(f"Ep {current_episode_global}/{total_planned_episodes} | "
                      f"Avg R: {avg_reward:.2f} | "
                      f"Win Rate: {win_rate*100:.1f}% | "
                      f"Eps: {eps_val:.3f}")

            if current_episode_global % save_interval == 0:
                agent.save(save_path_model)
                save_training_data(history, save_path_data)
        
        total_episodes_done += n_episodes
        phase_idx += 1
        
    agent.save(save_path_model)
    save_training_data(history, save_path_data)
    
    print("\n=== Final Evaluation ===")
    final_win_rate = evaluate_agent(env, agent, current_opponents, n_games_per_opponent=50)
    print(f"Final Win Rate (vs last pool): {final_win_rate*100:.1f}%")
    print("Training complete.")