import argparse
from corridor import Corridor
from models import SarsaAgent, GreedyPathAgent, RandomAgent
import os

def main():
    parser = argparse.ArgumentParser(description="Train RL Agent for Corridor")
    parser.add_argument("--model", type=str, default="sarsa", choices=["sarsa"], help="Model to train")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of episodes")
    parser.add_argument("--board_size", type=int, default=5, help="Board size (N)")
    parser.add_argument("--walls", type=int, default=6, help="Walls per player (reduced for 7x7)")
    parser.add_argument("--adversary", type=str, default="greedy", choices=["greedy", "random"], help="Adversary type")
    parser.add_argument("--save_path", type=str, default="saved_models/sarsa_model.pkl", help="Path to save model")
    
    args = parser.parse_args()
    
    # Setup Environment
    # 7x7 board usually has fewer walls than 9x9 (10 walls). 
    # Let's stick to user request "DO everything on 7x7 board size".
    # Default walls for 9x9 is 10. For 7x7 maybe 6 or 8?
    # I'll use args.walls.
    env = Corridor(N=args.board_size, walls_per_player=args.walls)
    
    # Setup Adversary
    if args.adversary == "greedy":
        # wall_prob=0.1 makes it occasionally place walls, making it harder than pure pathing
        adversary = GreedyPathAgent(name="Greedy", wall_prob=0.1) 
    else:
        adversary = RandomAgent(name="Random")
        
    # Setup Agent
    if args.model == "sarsa":
        agent = SarsaAgent(
            name="Sarsa", 
            alpha=0.1, 
            gamma=0.995, 
            epsilon=1.0, # Start with full exploration
            training_mode=True
        )
        
        if os.path.exists(args.save_path):
            print(f"Loading existing model from {args.save_path}...")
            agent.load(args.save_path)
            
        # Curriculum Training
        
        # Phase 1: Train against Random Agent (20% of episodes)
        print("\n=== Phase 1: Training against Random Agent ===")
        random_adversary = RandomAgent(name="Random")
        agent.train(env, random_adversary, int(args.episodes * 0.2), args.save_path)
        
        # Phase 2: Train against Greedy Agent (50% of episodes)
        print("\n=== Phase 2: Training against Greedy Agent ===")
        greedy_adversary = GreedyPathAgent(name="Greedy", wall_prob=0.1)
        # Boost exploration slightly for new adversary
        agent.epsilon = max(agent.epsilon, 0.3)
        agent.train(env, greedy_adversary, int(args.episodes * 0.5), args.save_path)

        # Phase 3: Train against Self (30% of episodes)
        print("\n=== Phase 3: Training against Self ===")
        # Boost exploration slightly for self-play to find new strategies
        agent.epsilon = max(agent.epsilon, 0.2)
        agent.train(env, agent, int(args.episodes * 0.3), args.save_path)
        
if __name__ == "__main__":
    main()
