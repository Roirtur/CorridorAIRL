import random
import math
from typing import List, Tuple
from corridor import Corridor
from models import BaseAgent
from utils.saving import save_tabular_model, save_training_data

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
    while not done and steps < 1000: # Safety limit
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

def tabular_training_loop(
    env: Corridor,
    agent: BaseAgent,
    opponents_schedule: List[Tuple[BaseAgent, int]],
    save_path_model: str,
    save_path_data: str,
    eval_interval: int = 100,
    save_interval: int = 500,
    min_epsilon: float = 0.1,
    n_eval_games: int = 10
):
    """
    Training loop with curriculum learning.
    """
    
    total_episodes = 0
    history = {
        "rewards": [],
        "lengths": [],
        "cumulative_episodes": [],
        "win_rates": [],
        "eval_episodes": []
    }
    
    print(f"Starting training for {agent.name}...")
    
    for opponent, n_episodes in opponents_schedule:
        print(f"\nCurriculum Phase: vs {opponent.name} for {n_episodes} episodes")
        
        phase_rewards = []
        
        for i in range(n_episodes):
            # cosine (slow start, fast end)
            # resets at each curriculum phase to allow exploration against new opponent
            if hasattr(agent, "epsilon"):
                progress = i / n_episodes
                decayed_epsilon = math.cos(progress * math.pi / 2)
                agent.epsilon = max(min_epsilon, decayed_epsilon)

            agent_player = 1 if random.random() < 0.5 else 2
            
            stats = agent.run_episode(env, opponent, agent_player=agent_player)

            reward = stats["reward"]
            steps = stats["steps"]
            
            history["rewards"].append(reward)
            history["lengths"].append(steps)
            history["cumulative_episodes"].append(total_episodes + i + 1)
            
            phase_rewards.append(reward)
            
            if (i + 1) % eval_interval == 0:
                avg_reward = sum(phase_rewards[-eval_interval:]) / eval_interval
                epsilon = getattr(agent, 'epsilon', 0.0)
                
                # Evaluation
                old_epsilon = getattr(agent, 'epsilon', 0.0)
                old_training_mode = getattr(agent, 'training_mode', True)
                
                # Set to greedy/eval mode
                agent.epsilon = 0.0
                agent.training_mode = False
                
                wins = 0
                for _ in range(n_eval_games):
                    wins += run_eval_game(env, agent, opponent)
                
                win_rate = wins / n_eval_games
                
                # Restore training mode
                agent.epsilon = old_epsilon
                agent.training_mode = old_training_mode
                
                history["win_rates"].append(win_rate)
                history["eval_episodes"].append(total_episodes + i + 1)
                
                print(f"Episode {i+1}/{n_episodes} | Avg Reward: {avg_reward:.3f} | Epsilon: {epsilon:.3f} | Win Rate: {win_rate*100:.1f}%")
                
            if (i + 1) % save_interval == 0:
                save_tabular_model(agent, save_path_model)
                save_training_data(history, save_path_data)
                
        total_episodes += n_episodes
        
    save_tabular_model(agent, save_path_model)
    save_training_data(history, save_path_data)
    print("Training complete.")