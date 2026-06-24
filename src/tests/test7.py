from model import SocialNetwork, Agent
from controller import Controller
import numpy as np


def test7():    
    n_agents = 4
    
    agent0 = Agent(
        index=0,
        opinion=0.95,
        influence_of_others=np.array([0.2, 0, 0, 0.8], dtype=np.float64),
    )
    agent1 = Agent(
        index=1,
        opinion=0.8,
        influence_of_others=np.array([0, 0.5, 0.5, 0], dtype=np.float64),
    )

    agent2 = Agent(
        index=2,
        opinion=0.20,
        influence_of_others=np.array([0, 0.95, 0.05, 0], dtype=np.float64),
    )
    agent3 = Agent(
        index=3,
        opinion=0.05,
        influence_of_others=np.array([0.25, 0, 0, 0.75], dtype=np.float64),
    )   

    social_network = SocialNetwork(agents=[agent0, agent1, agent2, agent3])
    social_network.set_influence_functions_of_agents(None)
    controller = Controller(social_network=social_network, n_iterations=8)
    controller.run_simulation()
    # controller.display_network_graphs_animation(include_self_loops=True)
    controller.plot_opinion_history()
    # print(controller.get_last_opinion_vector())