import argparse
import glob
import os
import re
from corridor import Corridor
from models import SarsaAgent, GreedyPathAgent, RandomAgent, DQNAgent, QlearningAgent
from models.rl_utils import generate_save_path

# Define available models to train
MODELS = {
    'sarsa': SarsaAgent,
    'dqn': DQNAgent,
    'qlearn': QlearningAgent
}

# Define available adversaries (factories to create fresh instances)
ADVERSARIES = {
    "greedy": lambda: GreedyPathAgent(name="Greedy", wall_prob=0.1),
    "random": lambda: RandomAgent(name="Random"),
    "sarsa": lambda: SarsaAgent(name="Sarsa_Adv", epsilon=0.01, training_mode=False),
    "qlearn": lambda: QlearningAgent(name="Qlearn_Adv", epsilon=0.01, training_mode=False)
}

def extract_model_info(filepath: str) -> dict:
    """Extract model info from filename: model_N{size}_E{episodes}_vs_{adversaries}.pkl"""
    basename = os.path.basename(filepath)
    
    # Pattern: model_N9_E10000_vs_random_greedy_qlearn_sarsa_self.pkl
    pattern = r"(\w+)_N(\d+)_E(\d+)_vs_(.+)\.pkl"
    match = re.match(pattern, basename)
    
    if match:
        model_name = match.group(1)
        board_size = int(match.group(2))
        episodes = int(match.group(3))
        adversaries = match.group(4).split('_')
        
        return {
            'model': model_name,
            'board_size': board_size,
            'episodes': episodes,
            'adversaries': set(adversaries),
            'path': filepath
        }
    return None

def find_models_with_params(board_size: int, adversaries: list, episodes: int) -> dict:
    """
    Find trained SARSA and QLEARN models that were trained on the same board size, adversaries, AND episodes.
    Returns: {'sarsa': path, 'qlearn': path} or {} if not found
    """
    found_models = {}
    adversaries_set = set(adversaries)
    if "self" in adversaries_set:
        adversaries_set.remove("self")
    adversaries_set.add("self")  # Always include self
    
    pattern = "saved_models/*.pkl"
    files = glob.glob(pattern)
    
    for filepath in files:
        info = extract_model_info(filepath)
        if not info:
            continue
        
        # Check if board size matches
        if info['board_size'] != board_size:
            continue
        
        # Check if episodes match
        if info['episodes'] != episodes:
            continue
        
        # Check if adversaries match (same set)
        if info['adversaries'] == adversaries_set:
            model_type = info['model']
            if model_type not in found_models:
                found_models[model_type] = filepath
                print(f"✓ Found {model_type.upper()} trained on N={board_size}, E={episodes} with adversaries {adversaries_set}: {filepath}")
    
    return found_models

def find_trained_model(model_name: str, board_size: int) -> str:
    """Find a trained model for the given type and board size"""
    pattern = f"saved_models/{model_name}_N{board_size}_*.pkl"
    files = glob.glob(pattern)
    if files:
        # Return the most recent file
        latest = max(files, key=os.path.getctime)
        return latest
    return None

def main():
    parser = argparse.ArgumentParser(description="Train RL Agent for Corridor")
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()), help="Model to train")
    parser.add_argument("--episodes", type=int, default=10000, help="Number of episodes")
    parser.add_argument("--board_size", type=int, default=9, help="Board size (N)")
    parser.add_argument("--walls", type=int, default=10, help="Walls per player (standard 10 for 9x9)")
    parser.add_argument("--adversary", nargs='+', default=None, help="Adversary type(s)")
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

    # Auto-detect trained models if no adversary specified
    if args.adversary is None:
        args.adversary = ["random", "greedy"]  # Default base adversaries
        
        # Search for trained models with the same board_size, episodes AND adversaries
        found_models = find_models_with_params(args.board_size, args.adversary, args.episodes)
        
        if found_models:
            for model_type, path in found_models.items():
                args.adversary.append(model_type)
                adversary_paths[model_type] = path
        
        if len(args.adversary) == 2:  # Only random and greedy
            print(f"No trained models found for N={args.board_size}, E={args.episodes} with these adversaries. Using Random + Greedy only.")

    # Generate default save path if not provided
    if args.save_path is None:
        # Create a descriptive name based on adversaries
        adv_names = list(args.adversary)
        if "self" not in adv_names:
            adv_names.append("self")
        adv_str = "_".join(adv_names)
        args.save_path = generate_save_path("saved_models", args.model, args.board_size, args.episodes, adv_str)
    
    print(f"\nInitializing training on {args.board_size}x{args.board_size} board with {args.walls} walls.")
    print(f"Optimizations enabled: Canonical State (Symmetry+Perspective), Action Pruning, Reward Shaping.")
    print(f"Adversaries: {args.adversary} + self")
    print(f"Save path: {args.save_path}\n")
    
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
            adversary = agent  # Self-play
        elif adv_name in ADVERSARIES:
            adversary = ADVERSARIES[adv_name]()
            
            # Load pretrained model if specified or found
            adv_path = adversary_paths.get(adv_name)
            if adv_path and os.path.exists(adv_path):
                print(f"Loading {adv_name} from {adv_path}")
                if hasattr(adversary, 'load'):
                    adversary.load(adv_path)
                    print(f"  ✓ Q-table loaded ({len(adversary.q_table)} states)")
                else:
                    print(f"  ⚠ Warning: {adv_name} does not support loading.")
            elif adv_name in ["sarsa", "qlearn"]:
                print(f"  ⚠ Warning: No trained {adv_name} model found. Will play randomly.")
        else:
            print(f"Unknown adversary: {adv_name}, skipping.")
            continue
            
        active_adversaries.append(adversary)
        
    if not active_adversaries:
        print("No valid adversaries found. Exiting.")
        return

    print(f"\n=== Training against: {', '.join([adv.name for adv in active_adversaries])} ===\n")
    agent.train(env, active_adversaries, args.episodes, args.save_path, 
                start_epsilon=args.epsilon, 
                end_epsilon=args.min_epsilon, 
                alpha=args.alpha,
                gamma=args.gamma,
                epsilon_decay=args.epsilon_decay)

if __name__ == "__main__":
    main()

