import random
from typing import List, Tuple
from corridor import Corridor
from models.base_agent import BaseAgent
from utils.saving import save_tabular_model, save_training_data

def tabular_training_loop(
    env: Corridor,
    agent: BaseAgent,
    opponents_schedule: List[Tuple[BaseAgent, int]],
    save_path_model: str,
    save_path_data: str,
    eval_interval: int = 50,
    save_interval: int = 250
):
    """
    Training loop with curriculum learning.
    """
    
    total_episodes = 0
    history = {
        "rewards": [],
        "lengths": [],
        "cumulative_episodes": []
    }
    
    print(f"Starting training for {agent.name}...")
    
    for opponent, n_episodes in opponents_schedule:
        print(f"\nCurriculum Phase: vs {opponent.name} for {n_episodes} episodes")
        
        phase_rewards = []
        
        for i in range(n_episodes):
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
                print(f"Episode {i+1}/{n_episodes} | Avg Reward: {avg_reward:.3f} | Epsilon: {epsilon:.3f}")
                
            if (i + 1) % save_interval == 0:
                save_tabular_model(agent, save_path_model)
                save_training_data(history, save_path_data)
                
        total_episodes += n_episodes
        
    save_tabular_model(agent, save_path_model)
    save_training_data(history, save_path_data)
    print("Training complete.")