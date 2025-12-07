from typing import List
from corridor import Corridor
import time

# =======================
# 1) Interface d'agent
# =======================

from models import *

# =======================
# 2) Boucle de partie
# =======================

def play_game(env: Corridor, agent1: BaseAgent, agent2: BaseAgent, render: bool = False, max_moves: int = 500) -> dict:
    obs = env.reset()
    if render:
        env.render()

    agents = {1: agent1, 2: agent2}
    history: List[dict] = []

    for _ in range(max_moves):
        player = obs["to_play"]
        agent = agents[player]
        action = agent.select_action(env, obs)
        obs, reward, done, info = env.step(action)

        history.append({
            "player": player,
            "action": action,
            "reward": reward,
            "done": done,
            "info": info
        })

        if render:
            env.render()

        if done:
            winner = info.get("winner")
            return {
                "winner": winner,
                "move_count": env.move_count,
                "history": history
            }

    # Sécurité: match nul si trop long
    return {"winner": None, "move_count": env.move_count, "history": history}

def train(env: Corridor, agent1: QlearningAgent, agent2: BaseAgent, nb_epochs: int):
    start = time.time()
    agents = {1: agent1, 2: agent2}
    for epoch in range(nb_epochs):
        obs = env.reset()
        done = False
        last_p1_state = None
        last_p1_action = None

        for _ in range(500):
            player = obs["to_play"]
            agent = agents[player]
            action = agent.select_action(env, obs)
            state_before = obs
            d_before = env.shortest_path_length(1) or 0
            obs, reward, done, info = env.step(action)
            d_after = env.shortest_path_length(1) or 0
            if player == 1:
                reward += 0.05 * (d_before - d_after)  # bonus si on se rapproche du but

            if player == 1 and isinstance(agent1, QlearningAgent):
                # P1 vient de jouer : on stocke et on met à jour normalement
                last_p1_state = state_before
                last_p1_action = action
                agent1.update(state_before, action, reward, obs, done)

            if done:
                winner = info.get("winner")
                # Si P2 gagne, on pénalise le dernier coup de P1
                if winner == 2 and isinstance(agent1, QlearningAgent) and last_p1_state and last_p1_action:
                    agent1.update(last_p1_state, last_p1_action, -1.0, obs, True)
                break

        # Si nul (500 coups), on termine sur reward 0 pour le dernier coup de P1
        if not done and isinstance(agent1, QlearningAgent) and last_p1_state and last_p1_action:
            agent1.update(last_p1_state, last_p1_action, 0.0, obs, True)

        # Décroissance d’epsilon une fois par épisode
        if isinstance(agent1, QlearningAgent):
            agent1.epsilon = max(agent1.min_epsilon, agent1.epsilon * agent1.epsilon_decay)

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: epsilon = {agent1.epsilon:.4f}, Q-table size = {len(agent1.q_table)}")

    elapsed = time.time() - start
    print("Training of agent ", agent1.name, " against ", agent2.name, " done in ", elapsed, " seconds")
    
def evaluate(agent1: BaseAgent, n_games: int = 50, render: bool = False):
    env = Corridor(N=9, walls_per_player=10)
    agent2 = GreedyPathAgent(name="Greedy-2", wall_prob=0.0, seed=321)

    results = {"P1": 0, "P2": 0, "Draw": 0}
    for g in range(n_games):
        out = play_game(env, agent1, agent2, render=render)
        winner = out["winner"]
        if winner == 1:
            results["P1"] += 1
        elif winner == 2:
            results["P2"] += 1
        else:
            results["Draw"] += 1

    total = n_games
    print(f"\n=== Evaluation over {total} games ===")
    print(f"P1 wins: {results['P1']}")
    print(f"P2 wins: {results['P2']}")
    print(f"Draws  : {results['Draw']}")
    print("====================================")
   
# def evaluate(agent1: BaseAgent, n_games: int = 50, render: bool = False):
#     env = Corridor(N=9, walls_per_player=10)

#     # Remplace RandomAgent par votre agent :
#     #agent1 = RandomAgent(name="Random-1", seed=123)
#     agent2 = GreedyPathAgent(name="Greedy-2", wall_prob=0.0, seed=321)

#     results = {"P1": 0, "P2": 0, "Draw": 0}
#     for g in range(n_games):
#         # Alterner qui commence
#         if g % 2 == 0:
#             # P1 = agent1, P2 = agent2
#             pass
#         else:
#             # On échange les noms pour garder affichage cohérent
#             agent1, agent2 = agent2, agent1

#         out = play_game(env, agent1, agent2, render=render)
#         winner = out["winner"]
#         if winner == 1:
#             results["P1"] += 1
#         elif winner == 2:
#             results["P2"] += 1
#         else:
#             results["Draw"] += 1

#         # Ré-inverse pour la prochaine itération si on avait inversé
#         if g % 2 == 1:
#             agent1, agent2 = agent2, agent1

#     total = n_games
#     print(f"\n=== Evaluation over {total} games ===")
#     print(f"P1 wins: {results['P1']}")
#     print(f"P2 wins: {results['P2']}")
#     print(f"Draws  : {results['Draw']}")
#     print("====================================")


if __name__ == "__main__":
    # Lancer une partie unique avec rendu:

    q_agent = QlearningAgent(
    gamma=0.99,           # Facteur de discount (0.95-0.99 classique)
    alpha=0.5,            # Taux d'apprentissage (0.01-0.2)
    epsilon=1.0,          # Exploration initiale (100%)
    epsilon_decay=0.995,  # Décroissance après chaque épisode (0.99-0.999)
    min_epsilon=0.01,     # Exploration minimale (1-5%)
    name="QLearningAgent",
    seed=42
    )

    #play_game(Corridor(), q_agent, GreedyPathAgent(), render=True)
    evaluate(agent1=q_agent, n_games = 10, render = False)
    train(Corridor(), q_agent, GreedyPathAgent(), nb_epochs=500)
    evaluate(agent1=q_agent, n_games = 10, render = False)
    # Lancer une évaluation
    # evaluate(n_games=20, render=False)