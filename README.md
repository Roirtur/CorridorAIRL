# Corridor RL Agent


This project implements a modular framework for training and evaluating multiple reinforcement learning agents for the Corridor/Quoridor game. The goal is to develop agents that learn to play strategically and win against various types of opponents, using both tabular and deep RL methods.


## Project Structure

```
.
├── corridor.py              # Game environment (Corridor/Quoridor rules)
├── corridor_starter.py      # Legacy starter script
├── start_training.py        # Interactive training script
├── start_evaluation.py      # Interactive evaluation script
├── experiments.ipynb        # Jupyter notebook for experiments and analysis
├── requirements.txt         # Python dependencies
├── models/                  # Agent implementations
│   ├── __init__.py
│   ├── base_agent.py        # Base agent interface (all agents inherit)
│   ├── random/
│   │   └── random_agent.py  # Random baseline agent
│   ├── greedy/
│   │   └── greedy_path_agent.py  # Greedy heuristic agent
│   ├── qlearning/
│   │   └── qlearning_agent.py    # Q-Learning tabular agent
│   ├── sarsa/
│   │   └── sarsa_agent.py        # SARSA tabular agent
│   └── dqn/
│       ├── dqn_agent.py          # Deep Q-Network agent
│       ├── dqn_network.py        # DQN neural network architecture
│       ├── action_encoder.py     # Action encoding for DQN
│       └── prioritized_replay.py # Experience replay buffer
├── utils/
│   ├── representation.py    # State representation functions
│   ├── saving.py            # Model saving/loading utilities
│   └── training.py          # Training loop utilities
├── saved_models/            # Trained model checkpoints (auto-named)
└── report/
   └── report.tex           # LaTeX project report
```

## Installation


Install the required dependencies:

```bash
pip install -r requirements.txt
```

Dependencies include:
- `numpy` - Numerical computations
- `matplotlib` - Plotting and visualization
- `torch` - Deep learning (for DQN agent)
- `notebook` - Jupyter notebook support

## Training Agents


Use the interactive training script to train Q-Learning, SARSA, or DQN agents:

```bash
python start_training.py
```

The script will guide you through:
1. **Agent Selection**: Choose Q-Learning, SARSA, or DQN (Deep Q-Network)
2. **Board Size**: Set the game board dimensions (default: 5x5)
3. **Training Episodes**: Number of episodes to train
4. **Opponent Schedule**: Choose training curriculum:
   - Random agent only
   - Random to Greedy (Random → Greedy)
   - Mixed curriculum (e.g., Random → Greedy → Mixed Pool)

Models and training data are automatically saved to `saved_models/` with descriptive names like:
- `QAgent_B5_E5000_VSCurriculumMixed.pkl` (board size 5, 5000 episodes, curriculum)
- `SarsaAgent_B9_E10000_VSGreedy.pkl` (board size 9, 10000 episodes, greedy)
- `DQNAgent_B5_E2000_VSRandomToGreedy.pth` (DQN, board size 5, 2000 episodes, curriculum)

The naming format includes agent type, board size, episode count, and opponent curriculum for easy identification. Model saving/loading is handled by utility functions in `utils/saving.py`.

## Evaluating Agents


Use the interactive evaluation script to test trained agents:

```bash
python start_evaluation.py
```


The script provides:
1. **Model Selection**: Choose from saved models or provide a custom path
2. **Opponent Selection**: 
   - Random Agent
   - Greedy Agent
   - Another trained agent (auto-detected type)
3. **Number of Games**: Specify evaluation game count
4. **Board Visualization**: Choose how many final board states to display
5. **Starting Policy**:
   - Always Player 1 (your agent)
   - Always Player 2 (opponent)
   - Randomized
6. **Board Size**: Auto-detected from filename or manual entry

Results include win rate, loss rate, draw rate, average game length, and optional board visualizations showing the final game states with player roles clearly indicated. The evaluation script supports all agent types and can compare any combination of agents.

## Implemented Agents


### Tabular Agents
- **Q-Learning**: Off-policy TD learning with epsilon-greedy exploration
- **SARSA**: On-policy TD learning with epsilon-greedy exploration

Both use:
- State representation: Normalized board features
- Reward shaping: Small step penalties (-0.01), win (+1.0), loss (-1.0)
- Hyperparameters: α (learning rate), γ (discount), ε (exploration)

### Baseline Agents
- **Random Agent**: Uniform random action selection
- **Greedy Agent**: Heuristic shortest-path strategy

### Advanced Agents
- **DQN Agent**: Deep Q-Network with experience replay and prioritized replay buffer (advanced implementation, uses PyTorch)

All agents inherit from a common `BaseAgent` interface, making it easy to add new agent types or modify existing ones. The agent system is fully modular and extensible.

## Experiments


Explore training results and visualizations in:
```bash
jupyter notebook experiments.ipynb
```

## Project Report


The LaTeX report is located in `report/report.tex`. It includes:
- Learning methodology and algorithm descriptions
- Implementation details (state representation, rewards, hyperparameters)
- Experimental results with tables and graphs
- Analysis and discussion of agent performance
- Limitations and future work


## Game Rules

Corridor/Quoridor is a 2-player board game where:
- Each player starts on opposite sides of an N×N board
- Goal: Reach the opposite side first
- Players can either **move** their pawn or **place walls** to block the opponent
- Each player has a limited number of walls (default: 10)


## Results

Trained models demonstrate:
- Strong performance against random opponents
- Competitive play against greedy heuristic agents
- Strategic wall placement and movement decisions
- Curriculum and mixed-opponent training improves robustness and generalization

See `experiments.ipynb` and the report for detailed analysis.


## Authors

Niels ROUDEAU & Arthur MACDONALD


## License

This is an academic project for reinforcement learning coursework.
