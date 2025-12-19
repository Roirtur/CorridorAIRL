# Made with AI to fasten the task

import os
from corridor import Corridor
from models import DQNAgent, QlearningAgent, SarsaAgent, RandomAgent, GreedyPathAgent
from utils.saving import load_tabular_model, parse_model_info


def list_saved_models(directory: str = "saved_models") -> list:
    """Lists all model files in the specified directory."""
    if not os.path.exists(directory):
        return []
    # Include .pth files for DQN and .pkl for tabular
    files = [f for f in os.listdir(directory) if f.endswith('.pkl') or f.endswith('.pth')]
    return sorted(files)


def select_model(prompt: str = "Select a model:") -> str:
    """
    Allows user to select a model from saved_models or provide custom path.
    Returns the full path to the selected model.
    """
    print(f"\n{prompt}")
    print("1. Choose from saved_models/")
    print("2. Enter custom path")
    
    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            models = list_saved_models()
            if not models:
                print("No models found in saved_models/")
                print("Please enter a custom path.")
                choice = "2"
            else:
                print("\nAvailable models:")
                for i, model in enumerate(models, 1):
                    print(f"{i}. {model}")
                
                while True:
                    try:
                        idx = int(input(f"\nSelect model (1-{len(models)}): ").strip())
                        if 1 <= idx <= len(models):
                            return os.path.join("saved_models", models[idx - 1])
                        print(f"Invalid choice. Enter a number between 1 and {len(models)}.")
                    except ValueError:
                        print("Invalid input. Enter a number.")
        
        if choice == "2":
            while True:
                path = input("Enter model path: ").strip()
                if os.path.exists(path):
                    return path
                print(f"File not found: {path}")


def select_opponent() -> tuple:
    """
    Allows user to select an opponent agent.
    Returns (opponent_type, path_or_none)
    """
    print("\nSelect Opponent:")
    print("1. Random Agent")
    print("2. Greedy Agent")
    print("3. Agent (load from file)")
    
    while True:
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            return ("random", None)
        elif choice == "2":
            return ("greedy", None)
        elif choice == "3":
            path = select_model("Select opponent agent model:")
            return ("agent", path)
        else:
            print("Invalid choice. Please try again.")


def get_number_of_games() -> int:
    """Prompts user for number of games to play."""
    while True:
        try:
            n = int(input("\nEnter number of games to play (default 100): ").strip() or "100")
            if n < 1:
                print("Number of games must be positive.")
                continue
            return n
        except ValueError:
            print("Invalid input. Enter a number.")


def get_starting_policy() -> str:
    """
    Prompts user for starting policy.
    Returns 'player1', 'player2', or 'random'
    """
    print("\nSelect who starts:")
    print("1. Always Player 1 (loaded agent)")
    print("2. Always Player 2 (opponent)")
    print("3. Randomized")
    
    while True:
        choice = input("Enter choice (1-3): ").strip()
        if choice == "1":
            return "player1"
        elif choice == "2":
            return "player2"
        elif choice == "3":
            return "random"
        else:
            print("Invalid choice. Please try again.")


def load_agent_from_path(path: str, agent_type: str = None):
    """
    Loads an agent from a file path.
    Tries to infer agent type from filename if not provided.
    """
    # 1. Parse Info from Filename
    model_info = parse_model_info(path)
    
    # 2. Infer Agent Type if needed
    if agent_type is None:
        agent_name_lower = model_info['agent_name'].lower()
        
        if 'dqn' in agent_name_lower:
            agent_type = 'dqn'
        elif 'qlearn' in agent_name_lower or 'q_learn' in agent_name_lower or 'qagent' in agent_name_lower:
            agent_type = 'qlearning'
        elif 'sarsa' in agent_name_lower:
            agent_type = 'sarsa'
        else:
            # Ask user
            print(f"\nCannot infer agent type from filename: {os.path.basename(path)}")
            print("1. Q-Learning")
            print("2. SARSA")
            print("3. DQN")
            while True:
                choice = input("Enter agent type (1-3): ").strip()
                if choice == "1":
                    agent_type = 'qlearning'
                    break
                elif choice == "2":
                    agent_type = 'sarsa'
                    break
                elif choice == "3":
                    agent_type = 'dqn'
                    break
                print("Invalid choice.")
    
    # 3. Initialize and Load
    if agent_type == 'dqn':
        agent_name = f"LoadedDQNAgent"
        
        # DQN needs board size to build the network structure
        board_size = model_info.get('board_size')
        
        if board_size is None:
            print("\nDQN requires board size to initialize the network.")
            while True:
                try:
                    board_size = int(input("Enter board size (e.g. 5, 9): ").strip())
                    if board_size >= 3:
                        break
                except ValueError:
                    pass
                print("Invalid board size.")
        
        agent = DQNAgent(name=agent_name, board_size=board_size, training_mode=False)
        agent.load(path) # DQN has its own load method
        return agent

    else:
        # Tabular Agents
        if agent_type == 'qlearning':
            agent = QlearningAgent(name="LoadedQAgent", training_mode=False)
        elif agent_type == 'sarsa':
            agent = SarsaAgent(name="LoadedSarsaAgent", training_mode=False)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        load_tabular_model(agent, path)
        return agent


def run_evaluation(agent, opponent, env, n_games, starting_policy, n_boards_to_save=0):
    """
    Runs evaluation games and returns results.
    """
    wins = 0
    losses = 0
    draws = 0
    total_steps = 0
    final_boards = []  # Store final board states
    
    import random
    
    for game_idx in range(n_games):
        # Determine who starts this game
        if starting_policy == "player1":
            agent_player = 1
        elif starting_policy == "player2":
            agent_player = 2
        else:  # random
            agent_player = random.choice([1, 2])
        
        # Run game
        obs = env.reset()
        steps = 0
        max_steps = 250
        
        while steps < max_steps:
            current_player = obs["to_play"]
            
            if current_player == agent_player:
                action = agent.select_action(env, obs)
            else:
                action = opponent.select_action(env, obs)
            
            obs, _, done, info = env.step(action)
            steps += 1
            
            if done:
                winner = info.get("winner")
                if winner == agent_player:
                    wins += 1
                elif winner is None:
                    draws += 1
                else:
                    losses += 1
                total_steps += steps
                
                # Save board state if needed
                if len(final_boards) < n_boards_to_save:
                    final_boards.append({
                        "env": env.clone(),
                        "winner": winner,
                        "agent_player": agent_player,
                        "steps": steps,
                        "game_num": game_idx + 1
                    })
                break
        else:
            # Max steps reached
            draws += 1
            total_steps += steps
            
            # Save board state if timeout and needed
            if len(final_boards) < n_boards_to_save:
                final_boards.append({
                    "env": env.clone(),
                    "winner": None,
                    "agent_player": agent_player,
                    "steps": steps,
                    "game_num": game_idx + 1
                })
        
        # Progress indicator
        if (game_idx + 1) % 10 == 0 or game_idx == 0:
            print(f"Completed {game_idx + 1}/{n_games} games...", end="\r")
    
    print()  # New line after progress
    
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "total_games": n_games,
        "avg_steps": total_steps / n_games if n_games > 0 else 0,
        "final_boards": final_boards
    }


def main():
    print("=" * 50)
    print("   Corridor RL Agent Evaluation")
    print("=" * 50)
    
    # 1. Select main agent to evaluate
    print("\n=== Select Agent to Evaluate ===")
    agent_path = select_model("Select the agent model to evaluate:")
    agent = load_agent_from_path(agent_path)
    print(f"✓ Loaded agent: {agent.name}")
    
    # 2. Select opponent
    print("\n=== Select Opponent ===")
    opponent_type, opponent_path = select_opponent()
    
    if opponent_type == "random":
        opponent = RandomAgent()
        print(f"✓ Opponent: Random Agent")
    elif opponent_type == "greedy":
        opponent = GreedyPathAgent()
        print(f"✓ Opponent: Greedy Agent")
    elif opponent_type == "agent":
        opponent = load_agent_from_path(opponent_path)
        print(f"✓ Opponent: {opponent.name}")
    
    # 3. Get number of games
    n_games = get_number_of_games()
    
    # 3b. Get number of boards to display
    while True:
        try:
            n_boards_display = int(input("\nNumber of final boards to display (default 0): ").strip() or "0")
            if n_boards_display < 0:
                print("Number must be non-negative.")
                continue
            if n_boards_display > n_games:
                print(f"Cannot display more boards than games ({n_games}).")
                continue
            break
        except ValueError:
            print("Invalid input. Enter a number.")
    
    # 4. Get starting policy
    starting_policy = get_starting_policy()
    
    # 5. Board size
    # Try to reuse the board size from the loaded agent if possible (for DQN specifically)
    if hasattr(agent, "board_size"):
        suggested_board_size = agent.board_size
    else:
        # Fallback for tabular agents if not explicit
        model_info = parse_model_info(agent_path)
        suggested_board_size = model_info.get('board_size')
    
    if suggested_board_size:
        print(f"\nDetected board size {suggested_board_size} from agent/filename.")
        use_detected = input(f"Use board size {suggested_board_size}? (Y/n): ").strip().lower()
        if use_detected in ['', 'y', 'yes']:
            board_size = suggested_board_size
        else:
            while True:
                try:
                    board_size = int(input("Enter board size: ").strip())
                    if board_size < 3:
                        print("Board size must be at least 3.")
                        continue
                    break
                except ValueError:
                    print("Invalid input. Enter a number.")
    else:
        print("\nCould not detect board size automatically.")
        while True:
            try:
                board_size = int(input("Enter board size (default 5): ").strip() or "5")
                if board_size < 3:
                    print("Board size must be at least 3.")
                    continue
                break
            except ValueError:
                print("Invalid input. Enter a number.")
    
    # Initialize environment
    env = Corridor(N=board_size)
    
    # Run evaluation
    print("\n" + "=" * 50)
    print("   Starting Evaluation")
    print("=" * 50)
    print(f"Agent: {agent.name}")
    print(f"Opponent: {opponent.name}")
    print(f"Board: {board_size}x{board_size}")
    print(f"Games: {n_games}")
    print(f"Starting: {starting_policy}")
    print("=" * 50)
    
    results = run_evaluation(agent, opponent, env, n_games, starting_policy, n_boards_display)
    
    # Display results
    print("\n" + "=" * 50)
    print("   Evaluation Results")
    print("=" * 50)
    print(f"Total Games: {results['total_games']}")
    print(f"Wins:        {results['wins']} ({results['wins']/results['total_games']*100:.1f}%)")
    print(f"Losses:      {results['losses']} ({results['losses']/results['total_games']*100:.1f}%)")
    print(f"Draws:       {results['draws']} ({results['draws']/results['total_games']*100:.1f}%)")
    print(f"Avg Steps:   {results['avg_steps']:.1f}")
    print("=" * 50)
    
    # Display final board states if requested
    if n_boards_display > 0 and results['final_boards']:
        print("\n" + "=" * 50)
        print("   Final Board States")
        print("=" * 50)
        for board_info in results['final_boards']:
            game_num = board_info['game_num']
            winner = board_info['winner']
            agent_player = board_info['agent_player']
            steps = board_info['steps']
            board_env = board_info['env']
            
            print(f"\n--- Game {game_num} (Steps: {steps}) ---")
            print(f"Player 1: {'Agent' if agent_player == 1 else 'Opponent'}")
            print(f"Player 2: {'Agent' if agent_player == 2 else 'Opponent'}")
            
            if winner == 1:
                result = "Player 1 (Agent) WON" if agent_player == 1 else "Player 1 (Opponent) WON"
            elif winner == 2:
                result = "Player 2 (Agent) WON" if agent_player == 2 else "Player 2 (Opponent) WON"
            else:
                result = "DRAW/TIMEOUT"
            print(f"Result: {result}")
            
            board_env.render()
        print("=" * 50)


if __name__ == "__main__":
    main()