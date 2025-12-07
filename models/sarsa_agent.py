from corridor import Corridor, Action
from typing import Dict, List, Optional
import random
from .base_agent import BaseAgent
import numpy as np
from collections import defaultdict
import pickle
import os

def state_to_features(obs: Dict, env: Corridor) -> tuple:
    N = obs['N']
    player = obs['to_play']
    
    # 1. Player perspective (always "me" vs "opponent")
    if player == 1:
        my_pos = obs['p1']
        opp_pos = obs['p2']
        my_walls = obs['walls_left'][1]
    else:
        # Flip board vertically for P2
        my_pos = (N - 1 - obs['p2'][0], obs['p2'][1])
        opp_pos = (N - 1 - obs['p1'][0], obs['p1'][1])
        my_walls = obs['walls_left'][2]
    
    # 2. Horizontal symmetry (always use left half)
    if my_pos[1] > N // 2:
        my_pos = (my_pos[0], N - 1 - my_pos[1])
        opp_pos = (opp_pos[0], N - 1 - opp_pos[1])
    
    # Simplified state: (my_pos, opp_pos, has_walls)
    # This reduces state space significantly to allow faster convergence
    return (my_pos, opp_pos, my_walls > 0)

class SarsaAgent(BaseAgent):
    """Sarsa Agent"""
    def __init__(
        self, 
        name: str = "Sarsa", 
        seed: int | None = None,
        alpha: float = 0.1,      # Learning rate
        gamma: float = 0.995,    # Discount factor
        epsilon: float = 0.1,    # Exploration rate
        training_mode: bool = True,
        load_path: Optional[str] = None
    ):
        super().__init__(name=name, seed=seed)
        
        # Hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Q-table: maps (state_features, action) -> Q-value
        self.q_table = defaultdict(float)
        
        # For tracking
        self.training_mode = training_mode
        self.episodes_trained = 0
        
        if load_path:
            self.load(load_path)
            
        if not self.training_mode:
            self.epsilon = 0.0 # Greedy in evaluation

    def select_action(self, env: Corridor, obs: Dict) -> Action:
        legal_actions = env.legal_actions()
        if not legal_actions:
            return None

        # Epsilon-greedy
        if self.training_mode and np.random.random() < self.epsilon:
            return random.choice(legal_actions)
            
        # Get normalized state
        state_features = state_to_features(obs, env)

        # Greedy exploitation: choose action with highest Q-value
        best_q = -float('inf')
        best_actions = []
        
        for action in legal_actions:
            q_val = self.q_table[(state_features, action)]
            if q_val > best_q:
                best_q = q_val
                best_actions = [action]
            elif q_val == best_q:
                best_actions.append(action)
        
        if not best_actions:
            return random.choice(legal_actions)
            
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, next_action, done):
        """
        SARSA update: Q(S, A) <- Q(S, A) + alpha * [R + gamma * Q(S', A') - Q(S, A)]
        """
        current_q = self.q_table[(state, action)]
        
        if done:
            target = reward
        else:
            next_q = self.q_table[(next_state, next_action)]
            target = reward + self.gamma * next_q
            
        self.q_table[(state, action)] += self.alpha * (target - current_q)

    def train(self, env: Corridor, adversary: BaseAgent, num_episodes: int, save_path: Optional[str] = None):
        """
        Train the agent against an adversary.
        """
        print(f"Starting training for {num_episodes} episodes against {adversary.name}...")
        
        # Decay epsilon
        start_epsilon = self.epsilon
        end_epsilon = 0.1
        epsilon_decay = (start_epsilon - end_epsilon) / num_episodes
        
        wins = 0
        
        def get_potential(env_ref, my_id):
            # Calculate potential based on current board state
            # We want to maximize (OppDist - MyDist).
            opp_id = 2 if my_id == 1 else 1
            my_d = env_ref.shortest_path_length(my_id)
            opp_d = env_ref.shortest_path_length(opp_id)
            return 0.2 * opp_d - 1.0 * my_d

        for episode in range(1, num_episodes + 1):
            obs = env.reset()
            
            # Randomly choose if we are P1 or P2
            my_id = random.choice([1, 2])
            
            # If we are P2, adversary moves first
            if my_id == 2:
                opp_action = adversary.select_action(env, obs)
                obs, _, done, info = env.step(opp_action)
                if done: # Should not happen on first move
                    continue
            
            state = state_to_features(obs, env)
            action = self.select_action(env, obs)
            
            # Calculate initial potential
            phi_s = get_potential(env, my_id)
            
            done = False
            while not done:
                # 1. Execute our action
                obs, reward, done, info = env.step(action)
                
                if done:
                    # We won
                    real_reward = 1.0
                    self.update(state, action, real_reward, None, None, True)
                    wins += 1
                    break
                
                # 2. Opponent's turn
                opp_action = adversary.select_action(env, obs)
                obs, opp_reward, done, info = env.step(opp_action)
                
                if done:
                    # Opponent won -> We lost
                    real_reward = -1.0
                    self.update(state, action, real_reward, None, None, True)
                    break
                
                # 3. Prepare next step
                next_state = state_to_features(obs, env)
                next_action = self.select_action(env, obs)
                
                # 4. Update with Shaping
                # R_shaped = R + gamma * phi(s') - phi(s)
                phi_next_s = get_potential(env, my_id)
                shaped_reward = 0.0 + self.gamma * phi_next_s - phi_s
                
                self.update(state, action, shaped_reward, next_state, next_action, False)
                
                state = next_state
                action = next_action
                phi_s = phi_next_s
            
            # Decay epsilon
            if self.epsilon > end_epsilon:
                self.epsilon -= epsilon_decay
                
            self.episodes_trained += 1
            
            if episode % 100 == 0:
                print(f"Episode {episode}/{num_episodes} - Win Rate (last 100): {wins}% - Epsilon: {self.epsilon:.3f}")
                wins = 0
                
        if save_path:
            self.save(save_path, env, adversary)

    def save(self, path: str, env: Corridor, adversary: BaseAgent):
        data = {
            "q_table": dict(self.q_table),
            "episodes": self.episodes_trained,
            "board_size": env.N,
            "walls": env.walls_per_player,
            "adversary": adversary.name,
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "gamma": self.gamma
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Model saved to {path}")

    def load(self, path: str):
        if not os.path.exists(path):
            print(f"No model found at {path}")
            return
            
        with open(path, "rb") as f:
            data = pickle.load(f)
            
        self.q_table = defaultdict(float, data["q_table"])
        self.episodes_trained = data.get("episodes", 0)
        self.epsilon = data.get("epsilon", self.epsilon)
        self.alpha = data.get("alpha", self.alpha)
        self.gamma = data.get("gamma", self.gamma)
        
        print(f"Model loaded from {path}")
        print(f"  Episodes: {self.episodes_trained}")
        print(f"  Board Size: {data.get('board_size')}")
        print(f"  Adversary: {data.get('adversary')}")
        