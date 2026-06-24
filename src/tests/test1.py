from model import SocialNetwork
from controller import Controller
import numpy as np
from math import exp


def test1():
    def equal_influence_function(iteration: int, current_influence: float, own_index: int, other_agent_index: int, *args, **kwargs) -> float:
        """
        Adjust the influence of an agent so that it gradually converges to 1/n_agents over time.

        Parameters:
        - iteration (int): The current iteration number of the simulation.
        - current_influence (float): The current influence of another agent on the agent.
        - own_index (int): The index of the agent whose influence is being adjusted.
        - other_agent_index (int): The index of the other agent whose influence is being considered.
        - *args: Positional arguments (not used here).
        - **kwargs: Keyword arguments containing 'n_agents' and 'alpha'.
            - n_agents (int): Total number of agents in the social network.
            - alpha (float) is the adjustment rate (controls how quickly the influence converges to 1/n_agents).
        """
        n_agents = kwargs.get('n_agents')
        alpha = kwargs.get('alpha', 0.1)

        target_influence = 1.0 / n_agents
        adjustment = alpha * (target_influence - current_influence)
        return current_influence + adjustment
    
    n_agents = 8
    seed = 371504638

    influence_chage_functions = [
        lambda iteration, current_influence, own_index, other_agent_index, **kwargs: equal_influence_function(
            iteration=iteration, 
            own_index=own_index,
            other_agent_index=other_agent_index,
            current_influence=current_influence, 
            n_agents=n_agents,
            alpha=0.3
        ) 
        for _ in range(n_agents)
    ]
    influence_chage_functions = None

    gamma = 100

    def distribution_function(opinion_difference: float) -> float:
        # return 1.0 / (1.0 + gamma * opinion_difference**2)
        return exp(-gamma * opinion_difference)
    
    social_network = SocialNetwork.generate_random_social_network(n_agents=n_agents, seed=seed)
    social_network.set_influence_functions_of_agents(influence_chage_functions)
    social_network.set_distribution_function(distribution_function)
    controller = Controller(social_network=social_network, n_iterations=18)
    controller.run_simulation_v2()
    controller.display_network_graphs_animation(include_self_loops=True, filter_vertex=3)
    controller.plot_opinion_history()
    controller.print_last_influence_matrices()
    # print(controller.get_last_opinion_vector())
    # print(controller.get_first_opinion_vector())