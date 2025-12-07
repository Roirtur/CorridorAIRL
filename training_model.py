import argparse
from corridor import Corridor
from models import SarsaAgent, GreedyPathAgent, RandomAgent
from models.rl_utils import generate_save_path
import os

def main():
    parser = argparse.ArgumentParser(description="Train RL Agent for Corridor")
    parser.add_argument("--model", type=str, default="sarsa", choices=["sarsa"], help="Model to train")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of episodes")
    parser.add_argument("--board_size", type=int, default=9, help="Board size (N)")
    parser.add_argument("--walls", type=int, default=10, help="Walls per player (standard 10 for 9x9)")
    parser.add_argument("--adversary", nargs='+', default=["greedy", "random"], help="Adversary type(s)")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save model")
    
    args = parser.parse_args()
    
    # Generate default save path if not provided
    if args.save_path is None:
        # Create a descriptive name based on adversaries
        adv_names = list(args.adversary)
        if "self" not in adv_names:
            adv_names.append("self")
        adv_str = "_".join(adv_names)
        args.save_path = generate_save_path("saved_models", args.model, args.board_size, args.episodes, adv_str)
    
    print(f"Initializing training on {args.board_size}x{args.board_size} board with {args.walls} walls.")
    print(f"Optimizations enabled: Canonical State (Symmetry+Perspective), Action Pruning, Reward Shaping.")
    print(f"Adversaries: {args.adversary} + self")
    print(f"Save path: {args.save_path}")
    
    # Setup Environment
    env = Corridor(N=args.board_size, walls_per_player=args.walls)
    
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
        adversaries_list = list(args.adversary)
        if "self" not in adversaries_list:
            adversaries_list.append("self")
            
        # Distribute episodes among phases
        episodes_per_phase = args.episodes // len(adversaries_list)
        
        for i, adv_name in enumerate(adversaries_list):
            print(f"\n=== Phase {i+1}: Training against {adv_name.capitalize()} ===")
            
            if adv_name == "greedy":
                adversary = GreedyPathAgent(name="Greedy", wall_prob=0.1)
            elif adv_name == "random":
                adversary = RandomAgent(name="Random")
            elif adv_name == "self":
                adversary = agent # Self-play
                # Boost exploration slightly for self-play if it was decayed too much?
                # Or just let it continue decaying?
                # Let's ensure at least some exploration
                agent.epsilon = max(agent.epsilon, 0.2)
            else:
                print(f"Unknown adversary: {adv_name}, skipping.")
                continue
                
            agent.train(env, adversary, episodes_per_phase, args.save_path)

if __name__ == "__main__":
    main()

