from corridor import Corridor

def get_shaped_reward(env: Corridor, my_id: int) -> float:
    """
    Calculates potential-based reward shaping.
    Phi(s) = 0.2 * OppDist - 1.0 * MyDist
    """
    opp_id = 2 if my_id == 1 else 1
    my_d = env.shortest_path_length(my_id)
    opp_d = env.shortest_path_length(opp_id)
    return 0.2 * opp_d - 1.0 * my_d
