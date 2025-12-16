import argparse
import os
import random
import numpy as np
from corridor import Corridor
from models import SarsaAgent, GreedyPathAgent, RandomAgent, DQNAgent, QlearningAgent, MyAgent
from models.rl_utils import run_episode, save_model, generate_save_path

MODELS = {
    'sarsa': SarsaAgent,
    'dqn': DQNAgent,
    'qlearn': QlearningAgent,
    'my_agent': MyAgent
}

def evaluate(agent, env, adversaries, num_games=20):
    """Evaluate agent against a list of adversaries."""
    results = {}
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Greedy for evaluation
    
    for adv in adversaries:
        wins = 0
        total_steps = 0
        for _ in range(num_games):
            # Evaluation runs are not training runs
            win, steps = run_episode(env, agent, adv, training=False)
            wins += win
            total_steps += steps
        
        win_rate = wins / num_games
        avg_steps = total_steps / num_games
        results[adv.name] = {'win_rate': win_rate, 'avg_steps': avg_steps}
        
    agent.epsilon = original_epsilon
    return results

def train(agent, env, total_episodes, save_path, min_epsilon=0.1, epsilon_decay_ratio=0.9995):
    """
    Professional training loop with curriculum and proper decay.
    epsilon_decay_ratio: proportion of training over which to decay epsilon (default: 95% of episodes)
    """
    # Adversaries
    random_adv = RandomAgent(name="Random")
    greedy_adv = GreedyPathAgent(name="Greedy", wall_prob=0.1)
    
    # Curriculum phases
    phase1_end = int(total_episodes * 0.3)
    phase2_end = int(total_episodes * 0.6)
    
    # Epsilon decay - exponential decay for smoother, slower reduction
    start_epsilon = agent.epsilon
    decay_end_episode = int(total_episodes * epsilon_decay_ratio)
    # Calculate decay rate for exponential decay: eps = start * (decay_rate ^ episode)
    decay_rate = (min_epsilon / start_epsilon) ** (1.0 / decay_end_episode)
    
    best_win_rate = 0.0
    
    print(f"Starting Professional Training for {agent.name}")
    print(f"Total Episodes: {total_episodes}")
    print(f"Epsilon: {start_epsilon:.3f} -> {min_epsilon:.3f} over {decay_end_episode} episodes (exponential)")
    print(f"Curriculum: Random -> Greedy -> Self/Mix")
    
    for episode in range(1, total_episodes + 1):
        # 1. Select Adversary based on curriculum
        if episode <= phase1_end:
            adversary = random_adv
        elif episode <= phase2_end:
            adversary = greedy_adv
        else:
            # Mix: 20% Random, 30% Greedy, 50% Self
            r = random.random()
            if r < 0.2: adversary = random_adv
            elif r < 0.5: adversary = greedy_adv
            else: adversary = agent # Self-play
            
        # 2. Run Episode
        run_episode(env, agent, adversary, training=True)
        
        # 3. Decay Epsilon (exponential)
        if episode <= decay_end_episode:
            agent.epsilon = max(min_epsilon, start_epsilon * (decay_rate ** episode))
            
        # 4. Evaluate & Log
        if episode % 100 == 0:
            eval_results = evaluate(agent, env, [random_adv, greedy_adv])
            avg_win = sum(r['win_rate'] for r in eval_results.values()) / len(eval_results)
            
            print(f"Ep {episode}/{total_episodes} | Eps: {agent.epsilon:.3f} | "
                  f"Win Rates: Random={eval_results['Random']['win_rate']:.2f}, Greedy={eval_results['Greedy']['win_rate']:.2f} | "
                  f"Avg Steps: Random={eval_results['Random']['avg_steps']:.1f}, Greedy={eval_results['Greedy']['avg_steps']:.1f}")
            
            # Save if best (or just periodically)
            if episode % 500 == 0 and avg_win >= best_win_rate:
                best_win_rate = avg_win
                if save_path:
                    save_model(agent, save_path, env, "professional", episode)
                    print(f"  -> New Best Model Saved! ({best_win_rate:.2f})")

def main():
    parser = argparse.ArgumentParser(description="Professional RL Training")
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()))
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--board_size", type=int, default=9)
    parser.add_argument("--walls", type=int, default=10)
    parser.add_argument("--save_path", type=str, default=None)
    
    args = parser.parse_args()
    
    env = Corridor(N=args.board_size, walls_per_player=args.walls)
    
    agent_cls = MODELS[args.model]
    kwargs = {
        'name': args.model.capitalize(), 
        'training_mode': True,
        'epsilon': 1.0,  # Start with full exploration
    }
    if args.model in ['dqn', 'my_agent'] :
        kwargs['board_size'] = args.board_size
        
    agent = agent_cls(**kwargs)
    
    if not args.save_path:
        args.save_path = f"saved_models/{args.model}_E{args.episodes}_N{args.board_size}.pkl"
        
    train(agent, env, args.episodes, args.save_path)

if __name__ == "__main__":
    main()

