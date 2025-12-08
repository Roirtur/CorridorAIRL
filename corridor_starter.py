from typing import List
from corridor import Corridor

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


def evaluate(n_games: int = 50, render: bool = False):
    env = Corridor(N=5)
    # env = Corridor(N=9, walls_per_player=10)

    # Remplace RandomAgent par votre agent :
    agent1 = RandomAgent(name="Random-1", seed=123)
    agent2 = GreedyPathAgent(name="Greedy-2", wall_prob=0.0, seed=321)

    results = {"P1": 0, "P2": 0, "Draw": 0}
    for g in range(n_games):
        # Alterner qui commence
        if g % 2 == 0:
            # P1 = agent1, P2 = agent2
            pass
        else:
            # On échange les noms pour garder affichage cohérent
            agent1, agent2 = agent2, agent1

        out = play_game(env, agent1, agent2, render=render)
        winner = out["winner"]
        if winner == 1:
            results["P1"] += 1
        elif winner == 2:
            results["P2"] += 1
        else:
            results["Draw"] += 1

        # Ré-inverse pour la prochaine itération si on avait inversé
        if g % 2 == 1:
            agent1, agent2 = agent2, agent1

    total = n_games
    print(f"\n=== Evaluation over {total} games ===")
    print(f"P1 wins: {results['P1']}")
    print(f"P2 wins: {results['P2']}")
    print(f"Draws  : {results['Draw']}")
    print("====================================")


if __name__ == "__main__":
    # Lancer une partie unique avec rendu:
    # Try to load trained model if available
    # We look for the curriculum trained model on 5x5 board
    model_path = "saved_models/sarsa_N6_E4000_vs_random_greedy_self.pkl"
    
    agent = SarsaAgent(training_mode=False, load_path=model_path)
    
    play_game(Corridor(N=6), GreedyPathAgent(), agent, render=True)

    # Lancer une évaluation
    # evaluate(n_games=20, render=False)