# Made with AI to fasten the task

import os
from corridor import Corridor
from models import QlearningAgent
from models import SarsaAgent
from models import RandomAgent
from models import GreedyPathAgent
from utils.training import training_loop
from utils.saving import generate_path_name

def get_user_input():
    print("=== Corridor RL Training Setup ===")
    
    # 1. Choose Agent
    print("\nSelect Agent:")
    print("1. Q-Learning")
    print("2. SARSA")
    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice == "1":
            agent_type = "qlearning"
            agent_name = input("Enter agent name (default: QAgent): ").strip() or "QAgent"
            break
        elif choice == "2":
            agent_type = "sarsa"
            agent_name = input("Enter agent name (default: SarsaAgent): ").strip() or "SarsaAgent"
            break
        print("Invalid choice. Please try again.")

    # 2. Board Size
    while True:
        try:
            board_size = int(input("\nEnter board size (default 5): ").strip() or "5")
            if board_size < 3:
                print("Board size must be at least 3.")
                continue
            break
        except ValueError:
            print("Invalid number.")

    # 3. Episodes
    while True:
        try:
            episodes = int(input("\nEnter total number of episodes (default 5000): ").strip() or "5000")
            if episodes < 1:
                print("Episodes must be positive.")
                continue
            break
        except ValueError:
            print("Invalid number.")

    # 4. Curriculum
    print("\nSelect Opponent Schedule:")
    print("1. Random Agent only")
    print("2. Greedy Agent only")
    print("3. Curriculum: Random (50%) -> Greedy (50%)")
    
    schedule = []
    opponent_str = ""
    
    while True:
        choice = input("Enter choice (1-3): ").strip()
        if choice == "1":
            schedule = [(RandomAgent(), episodes)]
            opponent_str = "Random"
            break
        elif choice == "2":
            schedule = [(GreedyPathAgent(), episodes)]
            opponent_str = "Greedy"
            break
        elif choice == "3":
            half = episodes // 2
            schedule = [
                (RandomAgent(), half),
                (GreedyPathAgent(), episodes - half)
            ]
            opponent_str = "Curriculum"
            break
        print("Invalid choice.")

    return agent_type, agent_name, board_size, episodes, schedule, opponent_str

def main():
    # Get parameters
    agent_type, agent_name, board_size, episodes, schedule, opponent_str = get_user_input()
    
    # Initialize Environment
    env = Corridor(N=board_size)
    
    # Initialize Agent
    if agent_type == "qlearning":
        agent = QlearningAgent(name=agent_name)
    else:
        agent = SarsaAgent(name=agent_name)
        
    # Generate Paths
    model_path = generate_path_name(agent_name, episodes, opponent_str, "model", board_size)
    data_path = generate_path_name(agent_name, episodes, opponent_str, "data", board_size)
    
    print(f"\n=== Starting Training ===")
    print(f"Agent: {agent.name} ({agent_type})")
    print(f"Board: {board_size}x{board_size}")
    print(f"Episodes: {episodes}")
    print(f"Opponent: {opponent_str}")
    print(f"Saving to: {model_path}")
    
    # Run Training
    training_loop(
        env=env,
        agent=agent,
        opponents_schedule=schedule,
        save_path_model=model_path,
        save_path_data=data_path,
    )

if __name__ == "__main__":
    main()
