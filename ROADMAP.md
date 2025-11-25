# Project Roadmap

This roadmap outlines the key steps for completing the Corridor RL agent project.

## Part 1: Agent Implementation

- [ ] **Define State and Action Representation:**
  - [ ] Determine how to represent the game board, player positions, and walls.
  - [ ] Define the set of possible actions the agent can take (move or place a wall).

- [ ] **Choose and Implement a Learning Algorithm:**
  - [ ] Select a suitable reinforcement learning algorithm (e.g., Q-learning, SARSA).
  - [ ] Implement the chosen algorithm in `models/my_agent.py`.

- [ ] **Choose an Exploration Policy:**
  - [ ] Implement an exploration-exploitation strategy (e.g., ε-greedy).

- [ ] **Tune Hyperparameters:**
  - [ ] Experiment with different values for the discount factor (γ).
  - [ ] Adjust the learning rate (α).
  - [ ] Fine-tune the exploration parameter (ε).

## Part 2: Evaluation

- [ ] **Set up Experiments:**
  - [ ] Modify `corridor_starter.py` to train your agent.
  - [ ] Implement evaluation matches for your agent against the random and heuristic agents.

- [ ] **Measure Performance:**
  - [ ] Calculate and record the win rate of your agent against opponents.
  - [ ] Measure the average length of the games.
  - [ ] Plot the learning curve to show the stability of the training process.

## Part 3: Reporting

- [ ] **Write the Report (`study/report.tex`):**
  - [ ] **Describe the Learning Method:** Explain the algorithm you chose and why.
  - [ ] **Implementation Choices:** Detail your state representation, reward function, and other key decisions.
  - [ ] **Experimental Results:** Present your findings using tables and graphs (win rate, game length, etc.).
  - [ ] **Critical Discussion:** Discuss the limitations of your agent and suggest potential improvements.

- [ ] **Final Submission:**
  - [ ] Ensure all source code for your agent is included.
  - [ ] Make sure the `corridor_starter.py` script can be used to demonstrate your agent.
  - [ ] Generate a PDF from your `report.tex` file.

## Optional Extensions

- [ ] **Implement a DQN Agent:**
  - [ ] Create a neural network to approximate the value function.
  - [ ] Train the DQN agent and compare its performance to the tabular agent.

- [ ] **Create Visualizations:**
  - [ ] Develop a way to visualize the agent's learned policies or game trajectories.
