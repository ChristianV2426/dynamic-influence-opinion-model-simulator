from model import SocialNetwork, Agent
from controller import Controller
import numpy as np


def test2():
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
    
    n_agents = 4

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
    
    agent0 = Agent(
        index=0,
        opinion=0.0,
        influence_of_others=np.array([0, 1, 0, 0], dtype=np.float64),
    )
    agent1 = Agent(
        index=1,
        opinion=0.2,
        influence_of_others=np.array([0.5, 0.5, 0, 0], dtype=np.float64),
    )

    agent2 = Agent(
        index=2,
        opinion=1.0,
        influence_of_others=np.array([0, 0, 0.2, 0.8], dtype=np.float64),
    )
    agent3 = Agent(
        index=3,
        opinion=0.8,
        influence_of_others=np.array([0, 0, 0.9, 0.1], dtype=np.float64),
    )   

    social_network = SocialNetwork(agents=[agent0, agent1, agent2, agent3])
    social_network.set_influence_functions_of_agents(influence_chage_functions)
    controller = Controller(social_network=social_network, n_iterations=10)
    controller.run_simulation()
    controller.display_network_graphs_animation(include_self_loops=True)
    controller.plot_opinion_history()
    