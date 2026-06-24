from model import SocialNetwork, Agent
from controller import Controller
import numpy as np
from math import exp


def test6():      
    agent0 = Agent(
        index=0,
        opinion=0.95,
        influence_of_others=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    )

    agent1 = Agent(
        index=1,
        opinion=0.80,
        influence_of_others=np.array([0.9, 0.095, 0.005, 0.0], dtype=np.float64),
    )

    agent2 = Agent(
        index=2,
        opinion=0.20, #0.75
        influence_of_others=np.array([0.0, 0.005, 0.095, 0.9], dtype=np.float64),
    )

    agent3 = Agent(
        index=3,
        opinion=0.05, #0.55
        influence_of_others=np.array([0.0, 0.0, 0.01, 0.99], dtype=np.float64),
    )

    gamma = 100

    def distribution_function(opinion_difference: float) -> float:
        # return 1.0 / (1.0 + gamma * opinion_difference)
        return exp(-gamma * opinion_difference**2)

    social_network = SocialNetwork(agents=[agent0, agent1, agent2, agent3], distribution_function=distribution_function)
    controller = Controller(social_network=social_network, n_iterations=1000)
    controller.run_simulation_v2()
    controller.display_network_graphs_animation(include_self_loops=True)
    controller.plot_opinion_history()
    controller.print_last_influence_matrices()
    # print(controller.get_last_opinion_vector())