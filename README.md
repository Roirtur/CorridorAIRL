# Corridor RL Agent

This project is an implementation of a reinforcement learning agent for the game of Corridor. The goal is to develop an agent that learns to play and win against other agents.

## Project Structure

```
.
├── corridor_starter.py   # Main script to run experiments
├── corridor.py           # Game logic (do not modify)
├── models/
│   ├── __init__.py
│   ├── base_agent.py
│   ├── greedy_path_agent.py # Heuristic agent
│   ├── my_agent.py          # Our agent implementation
│   └── random_agent.py      # Random agent
└── study/
    ├── experimentations.ipynb # Notebook for experiments
    └── report.tex             # LaTeX file for the report
```

## How to Run

To run the project, you can execute the `corridor_starter.py` script. You may need to install some dependencies first.

```bash
pip install -r requirements.txt
python corridor_starter.py
```

You can modify `corridor_starter.py` to train your agent and evaluate its performance against the provided agents.

## Roadmap

A detailed project plan is available in `ROADMAP.md`. It includes a checklist of tasks to guide through the implementation, evaluation, and reporting stages.

## Report

A 2-3 page report in PDF was generated. The LaTeX source file is `study/report.tex`. The report should include:

- A description of our learning method.
- Implementation choices (state representation, reward functions, etc.).
- Experimental results (tables, graphs, observations).
- A critical discussion of the limits and perspectives of our work.
