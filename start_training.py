# Made with AI to fasten the task

import os
from corridor import Corridor
from models import QlearningAgent
from models import SarsaAgent
from models import RandomAgent
from models import GreedyPathAgent
from models import DQNAgent
from utils.training import training_loop
from utils.saving import generate_path_name

def get_user_input():
    print("=== Corridor RL Training Setup ===")
    
    # 1. Choose Agent
    print("\nSelect Agent:")
    print("1. Q-Learning")
    print("2. SARSA")
    print("3. DQN (Deep Q-Network)")
    while True:
        choice = input("Enter choice (1-3): ").strip()
        if choice == "1":
            agent_type = "qlearning"
            agent_name = input("Enter agent name (default: QAgent): ").strip() or "QAgent"
            break
        elif choice == "2":
            agent_type = "sarsa"
            agent_name = input("Enter agent name (default: SarsaAgent): ").strip() or "SarsaAgent"
            break
        elif choice == "3":
            agent_type = "dqn"
            agent_name = input("Enter agent name (default: DQNAgent): ").strip() or "DQNAgent"
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
            if episodes < 100:
                print("Episodes must be at least 100.")
                continue
            break
        except ValueError:
            print("Invalid number.")

    print("\nSelect Training Curriculum:")
    print("1. Basic: Random Agent only")
    print("2. Intermediate: Random (50%) -> Greedy (50%)")
    print("3. Advanced: Random (30%) -> Greedy (30%) -> Mixed Pool (40%)")
    
    schedule = []
    opponent_str = ""
    
    while True:
        choice = input("Enter choice (1-3): ").strip()
        
        random_agent = RandomAgent()
        greedy_agent = GreedyPathAgent()
        
        if choice == "1":
            schedule = [(random_agent, episodes)]
            opponent_str = "RandomOnly"
            break
            
        elif choice == "2":
            half = episodes // 2
            schedule = [
                (random_agent, half),
                (greedy_agent, episodes - half)
            ]
            opponent_str = "RandomToGreedy"
            break
            
        elif choice == "3":
            # 30% Random, 30% Greedy, 40% Mixed (Random + Greedy)
            part1 = int(episodes * 0.3)
            part2 = int(episodes * 0.3)
            part3 = episodes - part1 - part2
            
            schedule = [
                (random_agent, part1),
                (greedy_agent, part2),
                ([random_agent, greedy_agent], part3) # List implies mixed pool
            ]
            opponent_str = "CurriculumMixed"
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
    elif agent_type == "sarsa":
        agent = SarsaAgent(name=agent_name)
    elif agent_type == "dqn":
        # DQN requires board_size to build the neural network input layer
        agent = DQNAgent(name=agent_name, board_size=board_size)
        
    # Generate Paths
    model_path = generate_path_name(agent_name, episodes, opponent_str, "model", board_size)
    data_path = generate_path_name(agent_name, episodes, opponent_str, "data", board_size)
    
    print(f"\n=== Starting Training Session ===")
    print(f"Agent:     {agent.name} ({agent_type})")
    print(f"Board:     {board_size}x{board_size}")
    print(f"Episodes:  {episodes}")
    print(f"Schedule:  {opponent_str}")
    print(f"Saving to: {model_path}")
    
    # Run Training
    training_loop(
        env=env,
        agent=agent,
        opponents_schedule=schedule,
        save_path_model=model_path,
        save_path_data=data_path,
        eval_interval=episodes // 20,
        save_interval=episodes // 5
    )

if __name__ == "__main__":
    main()