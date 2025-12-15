import argparse
from corridor import Corridor
from models import SarsaAgent, GreedyPathAgent, RandomAgent, DQNAgent
from models.rl_utils import generate_save_path
import os

# Define available models to train
MODELS = {
    'sarsa': SarsaAgent,
    'dqn': DQNAgent
}

# Define available adversaries (factories to create fresh instances)
ADVERSARIES = {
    "greedy": lambda: GreedyPathAgent(name="Greedy", wall_prob=0.1),
    "random": lambda: RandomAgent(name="Random"),
    "sarsa": lambda: SarsaAgent(name="Sarsa_Adv", epsilon=0.1, training_mode=False)
}

def main():
    parser = argparse.ArgumentParser(description="Train RL Agent for Corridor")
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()), help="Model to train")
    parser.add_argument("--episodes", type=int, default=10000, help="Number of episodes")
    parser.add_argument("--board_size", type=int, default=9, help="Board size (N)")
    parser.add_argument("--walls", type=int, default=10, help="Walls per player (standard 10 for 9x9)")
    parser.add_argument("--adversary", nargs='+', default=["random", "greedy"], help="Adversary type(s)")
    parser.add_argument("--load_adversary", nargs='+', help="Load adversary models: name=path")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save model")
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Starting epsilon")
    parser.add_argument("--gamma", type=float, default=0.98, help="Discount factor")
    parser.add_argument("--min_epsilon", type=float, default=0.1, help="Minimum epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=None, help="Epsilon decay factor (multiplicative). If not set, uses linear decay.")
    
    args = parser.parse_args()
    
    # Parse load_adversary
    adversary_paths = {}
    if args.load_adversary:
        for item in args.load_adversary:
            if '=' in item:
                key, val = item.split('=', 1)
                adversary_paths[key] = val
            else:
                print(f"Warning: Invalid format for load_adversary '{item}'. Expected name=path.")

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
    agent_cls = MODELS[args.model]
    agent = agent_cls(
        name=args.model.capitalize(),
        alpha=args.alpha, 
        gamma=args.gamma, 
        epsilon=args.epsilon, # Start with full exploration
        training_mode=True,
        board_size=args.board_size
    )
        
    if os.path.exists(args.save_path):
        print(f"Loading existing model from {args.save_path}...")
        agent.load(args.save_path)
        
    # Curriculum Training
    adversaries_list = list(args.adversary)
    if "self" not in adversaries_list:
        adversaries_list.append("self")
        
    active_adversaries = []
    
    for i, adv_name in enumerate(adversaries_list):
        if adv_name == "self":
            adversary = agent # Self-play
        elif adv_name in ADVERSARIES:
            adversary = ADVERSARIES[adv_name]()
            # Load pretrained model if specified
            if adv_name in adversary_paths:
                print(f"Loading {adv_name} from {adversary_paths[adv_name]}")
                if hasattr(adversary, 'load'):
                    adversary.load(adversary_paths[adv_name])
                else:
                    print(f"Warning: Adversary {adv_name} does not support loading.")
        else:
            print(f"Unknown adversary: {adv_name}, skipping.")
            continue
            
        active_adversaries.append(adversary)
        
    if not active_adversaries:
        print("No valid adversaries found. Exiting.")
        return

    print(f"\n=== Training against: {', '.join([adv.name for adv in active_adversaries])} ===")
    agent.train(env, active_adversaries, args.episodes, args.save_path, 
                start_epsilon=args.epsilon, 
                end_epsilon=args.min_epsilon, 
                alpha=args.alpha,
                gamma=args.gamma,
                epsilon_decay=args.epsilon_decay)

if __name__ == "__main__":
    main()

