from corridor import Corridor, Action
from typing import Dict, Optional
import random
from collections import defaultdict

from models import BaseAgent
from utils.representation import tabular_state_representation

class SarsaAgent(BaseAgent):
    """Sarsa Agent"""
    def __init__(
        self, 
        name: str = "Sarsa", 
        seed: int | None = None,
        alpha: float = 0.1,
        gamma: float = 0.995,
        epsilon: float = 0.1,
        training_mode: bool = True,
        load_path: Optional[str] = None
    ):
        super().__init__(name=name, seed=seed)
        
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.q_table = defaultdict(float)
        
        self.training_mode = training_mode

        if load_path:
            self.load(load_path)
            
        if not self.training_mode:
            self.epsilon = 0.0

    def select_action(self, env: Corridor, obs: Dict) -> Action:
        legal_actions = env.legal_actions()
        if not legal_actions:
            return None

        if self.training_mode and random.random() < self.epsilon:
            return random.choice(legal_actions)
            
        # normalized state
        state_features = tabular_state_representation(obs)

        # choose action with highest Q-value
        q_values = [self.q_table[(state_features, action)] for action in legal_actions]
        max_q = max(q_values)
        best_actions = [action for action, q in zip(legal_actions, q_values) if q == max_q]
        
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, next_action, done):
        current_q = self.q_table[(state, action)]
        
        if done:
            target = reward
        else:
            next_q = self.q_table[(next_state, next_action)]
            target = reward + self.gamma * next_q
            
        self.q_table[(state, action)] += self.alpha * (target - current_q)

    def run_episode(
        self,
        env: Corridor,
        opponent: BaseAgent,
        agent_player: int = 1,
        max_steps: int = 150
    ) -> Dict:
        """
        Run one SARSA training episode
        """
        obs = env.reset()
        
        episode_reward = 0
        steps = 0

        prev_state = None
        prev_action = None

        while steps < max_steps:
            current_player = obs["to_play"]
            
            if current_player == agent_player:
                agent = self
                is_learning = True
            else:
                agent = opponent
                is_learning = False

            # Current action and state
            current_action = agent.select_action(env, obs)
            current_state = tabular_state_representation(obs)

            # Execute action
            next_obs, _, done, info = env.step(current_action)


            if is_learning:
                reward = -0.01  # Small step penalty

                if prev_state is not None:
                    self.update(
                        prev_state,
                        prev_action,
                        reward,
                        current_state,
                        current_action,
                        False
                    )

                episode_reward += reward

                prev_state = current_state
                prev_action = current_action
            
            if done:
                if prev_state is not None:

                    winner = info.get("winner")
                    if winner == agent_player:
                        reward = 1.0  # Win
                    elif winner is None:
                        reward = -0.5  # Draw/timeout
                    else:
                        reward = -1.0  # Loss

                    self.update(
                        prev_state,
                        prev_action,
                        reward,
                        None,
                        None,
                        True
                    )

                    episode_reward += reward
                break

            obs = next_obs
            steps += 1
        
        return {
            "reward": episode_reward,
            "steps": steps,
        }