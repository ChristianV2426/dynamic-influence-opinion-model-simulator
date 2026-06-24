import networkx as nx
import numpy as np
from typing import List, Optional, Callable
from model import Agent


class SocialNetwork:
    def __init__(self, agents: List[Agent], distribution_function: Callable[[float], float] = None) -> None:
        """
        This class represents a social network of agents, where each agent has an opinion and influences others in the network.

        Attributes:
            agents (List[Agent]): A list of Agent objects representing the agents in the network.
            distribution_function (Callable[[float], float]): A function that defines how influence is distributed based on difference of opinions (homophily). This function takes a float (difference of opinion) and returns a float.
            n_agents (int): The total number of agents in the network. This is automatically determined from the length of the agents list.
            graph (nx.DiGraph): A directed graph representing the influence relationships between agents, constructed from the influence matrix. This is automatically created.
            node_positions (dict): A dictionary mapping agent indices to their positions in the graph layout for visualization purposes. This is automatically calculated using NetworkX's spring layout algorithm.  
        """
        self.agents = agents
        self.distribution_function = distribution_function

        self.n_agents = len(agents)
        self.graph = None
        self.node_positions = None

        self.sort_agents_by_index()
        self.init_opinion_vector()
        self.init_influence_matrix()
        self.update_graph()
        self.node_positions = nx.spring_layout(self.graph, seed=42)

    def sort_agents_by_index(self) -> None:
        self.agents.sort(key=lambda agent: agent.index)

    def init_opinion_vector(self) -> None:
        self.opinion_vector = np.zeros(self.n_agents, dtype=np.float64).reshape((-1, 1))
        self.opinion_vector[:, 0] = np.fromiter(
            (agent.get_opinion() for agent in self.agents),
            dtype=self.opinion_vector.dtype,
            count=self.n_agents            
        )

    def init_influence_matrix(self) -> None:
        self.influence_matrix = np.zeros((self.n_agents, self.n_agents), dtype=np.float64)
        self.influence_matrix[:] = np.vstack([agent.get_influence_of_others() for agent in self.agents])

    def update_graph(self) -> None:
        """
        This method creates a directed graph from the influence matrix. It's necessary to transpose the matrix because in DeGroot model, rows represent how an agent is influenced by others, while in NetworkX, edges go from influencer to influencee.
        """
        self.graph = nx.from_numpy_array(self.influence_matrix.T, create_using=nx.DiGraph)
        self.graph = nx.relabel_nodes(self.graph, lambda i: self.agents[i].index, copy=False)

    def get_influence_matrix(self) -> np.ndarray:
        return self.influence_matrix
    
    def get_opinion_vector(self) -> np.ndarray:
        return self.opinion_vector

    def get_graph(self) -> nx.DiGraph:
        return self.graph
    
    def get_node_positions(self) -> dict:
        return self.node_positions
    
    def set_influence_functions_of_agents(self, functions: List[Callable[..., float]]) -> None:
        for agent in self.agents:
            agent.set_influence_change_functions(functions)

    def set_distribution_function(self, distribution_function: Callable[[float], float]) -> None:
        self.distribution_function = distribution_function

    def update_opinions(self) -> None:
        """
        This method updates the opinion vector according to the DeGroot model:
        opinion_vector = influence_matrix @ opinion_vector
        After updating the opinion vector, which resides in the SocialNetwork, each agent is informed of its new opinion.
        """
        self.opinion_vector = self.influence_matrix @ self.opinion_vector

        # Let each agent know what their new opinion is
        for i, agent in enumerate(self.agents):
            agent.set_opinion(self.opinion_vector[i, 0])
            agent.add_iteration()
        
    def update_influences(self, **kwargs) -> None:
        """
        This method updates the influence matrix by asking each agent to update its influence_of_others vector.
        After updating each agent's influence_of_others, the influence matrix is reconstructed.

        This method is useful for recreating results such as those stated in the paper by Chatterjee and Seneta (1977) for "open-minded" agents.
        """
        for i, agent in enumerate(self.agents):
            agent.update_influence_of_others(**kwargs)
            self.influence_matrix[i, :] = agent.get_influence_of_others()

    def update_influences_v2(self, last_opinion_vector: np.ndarray) -> None:
        """
        This method updates the influence matrix by asking each agent to update its influence_of_others vector using the distribution function and the last opinion vector of all agents in the network.
        After updating each agent's influence_of_others, the influence matrix is reconstructed.

        This method is useful for implementing dynamic influence mechanisms driven by homophily.
        """
        for i, agent in enumerate(self.agents):
            agent.update_influence_of_others_v2(self.distribution_function, last_opinion_vector.flatten())
            self.influence_matrix[i, :] = agent.get_influence_of_others()   
    
    @staticmethod
    def generate_random_social_network(n_agents: int, seed: Optional[int] = None) -> 'SocialNetwork':
        if seed is None:
            seed = np.random.randint(0, 1_000_000_000)
            print(f"Generated seed for social network: {seed}")
            
        agents = [Agent.generate_random_agent(index=i, n_agents=n_agents, seed=seed) for i in range(n_agents)]
        return SocialNetwork(agents=agents)
        
