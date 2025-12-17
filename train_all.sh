#!/bin/bash

# train_all.sh - Script to train multiple RL models sequentially
# Usage: ./train_all.sh

set -e

echo "======================================"
echo "  Corridor RL - Model Training Suite"
echo "======================================"
echo ""

# Ask for number of models to train
read -p "How many models do you want to train? " num_models

# Validate input
if ! [[ "$num_models" =~ ^[0-9]+$ ]] || [ "$num_models" -lt 1 ]; then
    echo "Error: Please enter a valid positive number"
    exit 1
fi

# Arrays to store training configurations
declare -a models
declare -a episodes
declare -a board_sizes

# Default values
last_model="qlearn"
last_episodes="5000"
last_board_size="5"

echo ""
echo "Available models: sarsa, qlearn, dqn, my_agent"
echo ""

# Collect all training configurations
for ((i=1; i<=num_models; i++)); do
    echo "--- Configuration for Model #$i ---"
    
    # Model type
    read -p "Model type [default: $last_model]: " model
    model=${model:-$last_model}
    
    # Validate model type
    if [[ ! "$model" =~ ^(sarsa|qlearn|dqn|my_agent)$ ]]; then
        echo "Error: Invalid model type. Choose from: sarsa, qlearn, dqn, my_agent"
        exit 1
    fi
    
    # Number of episodes
    read -p "Number of episodes [default: $last_episodes]: " eps
    eps=${eps:-$last_episodes}
    
    # Validate episodes
    if ! [[ "$eps" =~ ^[0-9]+$ ]] || [ "$eps" -lt 1 ]; then
        echo "Error: Episodes must be a positive number"
        exit 1
    fi
    
    # Board size
    read -p "Board size (N x N) [default: $last_board_size]: " board
    board=${board:-$last_board_size}
    
    # Validate board size
    if ! [[ "$board" =~ ^[0-9]+$ ]] || [ "$board" -lt 3 ]; then
        echo "Error: Board size must be at least 3"
        exit 1
    fi
    
    # Store configuration
    models+=("$model")
    episodes+=("$eps")
    board_sizes+=("$board")
    
    # Update defaults for next iteration
    last_model="$model"
    last_episodes="$eps"
    last_board_size="$board"
    
    echo ""
done

# Display summary
echo "======================================"
echo "  Training Summary"
echo "======================================"
for ((i=0; i<num_models; i++)); do
    echo "Model #$((i+1)): ${models[$i]} - ${episodes[$i]} episodes - ${board_sizes[$i]}x${board_sizes[$i]} board"
done
echo "======================================"
echo ""

read -p "Start training? (y/n) " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

# Create saved_models directory if it doesn't exist
mkdir -p saved_models

# Start training
echo ""
echo "Starting training sequence..."
echo ""

for ((i=0; i<num_models; i++)); do
    model="${models[$i]}"
    eps="${episodes[$i]}"
    board="${board_sizes[$i]}"
    
    echo "======================================"
    echo "Training Model #$((i+1))/$num_models"
    echo "Model: $model | Episodes: $eps | Board: ${board}x${board}"
    echo "======================================"
    
    # Generate save path
    save_path="saved_models/${model}_E${eps}_N${board}.pkl"
    
    # Run training
    python3 training_model.py \
        --model "$model" \
        --episodes "$eps" \
        --board_size "$board" \
        --save_path "$save_path"
    
    echo ""
    echo "Model #$((i+1)) training completed!"
    echo "Saved to: $save_path"
    echo ""
done

echo "======================================"
echo "  All trainings completed!"
echo "======================================"
echo "Trained models saved in: saved_models/"
ls -lh saved_models/*.pkl 2>/dev/null || echo "No models found"