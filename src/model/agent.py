import numpy as np
from numpy.typing import NDArray
from typing import Callable, Optional, List


class Agent:
    def __init__(
            self, 
            index: int,
            opinion: float,
            influence_of_others: NDArray[np.float64],
            influence_change_functions: Optional[List[Callable[..., float]]] = None
            ) -> None:
        """
        This class represents an agent in a network with an opinion and influence on other agents.

        Attributes:
            index (int): The unique identifier for the agent.
            opinion (float): The current opinion of the agent, in the range [0.0, 1.0].
            influence_of_others (NDArray[np.float64]): A 1-D numpy array representing the influence of other agents on this agent. The sum of this array should be 1.0.
            influence_change_functions (Optional[List[Callable[..., float]]]): A list of functions that define how the influence of other agents changes over time. Each function corresponds to an agent in the network and takes parameters such as iteration number, own index, other agent index, current influence value, and any additional keyword arguments.
            iteration (int): The current iteration or time step in the simulation. Automatically initialized to 0.
            agents_in_network (int): The total number of agents in the network. Automatically determined from the length of the influence_of_others array.
        """
        self.index = index
        self.opinion = opinion
        self.influence_of_others = influence_of_others
        self.influence_change_functions = influence_change_functions

        self.iteration = 0
        self.agents_in_network = influence_of_others.shape[0]
        self.normalize_influence()

    def get_index(self) -> int:
        return self.index
    
    def get_opinion(self) -> float:
        return self.opinion
    
    def set_opinion(self, new_opinion: float) -> None:
        self.opinion = new_opinion

    def get_influence_of_others(self) -> NDArray[np.float64]:
        return self.influence_of_others

    def get_influence_change_functions(self) -> Optional[List[Callable[..., float]]]:
        return self.influence_change_functions
    
    def set_influence_change_functions(self, functions: List[Callable[..., float]]) -> None:
        self.influence_change_functions = functions

    def get_iteration(self) -> int:
        return self.iteration
    
    def add_iteration(self) -> None:
        self.iteration += 1

    def normalize_influence(self, epsilon: float = 1e-8) -> None:
        """
        This method ensures self.influence_of_others sums to 1.

        Assumptions:
        - self.influence_of_others is a 1-D numpy array of non-negative floats.
        - self.index is the agent's own position in that vector, and it is a valid index in self.influence_of_others.

        Steps:
        1) Compute total = sum(arr).
        2) If total is very close to 1 (within rel_tol), do nothing.
        3) If total > eps, divide the vector by total (this makes the sum exactly 1).
        4) If total <= eps (zero or near-zero), create a fallback vector where the agent has full influence on itself.
        """
        total = self.influence_of_others.sum()

        if abs(total - 1.0) <= epsilon:
            return
        
        elif total > epsilon:
            self.influence_of_others /= total
        
        else:
            fallback = np.zeros(self.agents_in_network, dtype=np.float64)
            fallback[self.index] = 1.0
            self.influence_of_others = fallback

    def update_influence_of_others(self, **kwargs) -> None:
        """
        This method updates the agent's influence_of_others vector by applying the corresponding influence change functions.

        This method is useful for recreating results such as those stated in the paper by Chatterjee and Seneta (1977) for "open-minded" agents.
        
        Steps:
        1) Check if a vector of influence change functions is provided. If not, do nothing.
        2) For each agent in the network, check if there is a corresponding influence change function (The vector may contain None entries, which indicates that the influence of that other agent on this agent never changes).
        3) If there is a function, update the influence of that agent on this agent by calling the function with the current iteration number, the current influence value, and any additional keyword arguments provided.
        4) After updating all influences, normalize the influence_of_others vector to ensure it sums to 1.
        """
        if self.influence_change_functions is not None:
            for i in range(self.agents_in_network):
                func = self.influence_change_functions[i]
                if func is not None:
                    self.influence_of_others[i] = func(
                        iteration=self.iteration,
                        own_index=self.index,
                        other_agent_index=i,
                        current_influence=self.influence_of_others[i],
                        **kwargs
                    )
            self.normalize_influence()

    def update_influence_of_others_v2(self, distribution_function: Callable[[float], float], last_opinion_vector: np.ndarray) -> None:
        """
        This method updates the agent's influence_of_others vector based on the distribution function and the last opinion vector of all agents in the network.
        
        This method is useful for implementing dynamic influence mechanisms driven by homophily.

        Steps:
        1) Check if the agent has all the influence on itself (i.e., influence_of_others[self.index] is close to 1). If so, skip the update to avoid unnecessary calculations.
        2) If a distribution function is provided, calculate a new distribution vector based on the opinion differences between this agent and all other agents, using the distribution function to determine how much influence to assign to each other agent.
        3) Normalize the new distribution vector to ensure it sums to 1, and set it as the new influence_of_others vector for this agent.
        """
        if (abs(self.influence_of_others[self.index] - 1.0)) <= 1e-4:
            return

        if distribution_function is not None:
            distribution_vector = np.zeros(self.agents_in_network, dtype=np.float64)

            for i in range(self.agents_in_network):
                if self.influence_of_others[i] != 0.0:
                    difference_opinion = abs(last_opinion_vector[i] - last_opinion_vector[self.index])
                    distribution_vector[i] = distribution_function(difference_opinion)

            self.influence_of_others = distribution_vector / distribution_vector.sum()
            self.normalize_influence()

    @staticmethod
    def generate_random_agent(index: int, n_agents: int, seed: Optional[int] = None) -> 'Agent':
        """
        This static method generates an agent with a random opinion and random influence_of_others vector.
        """
        if seed is not None:
            np.random.seed(seed + index)

        # The beta distribution generates probabilities values close to the extremes (near 0.0 and 1.0), resulting in polarized initial opinions.  
        opinion = np.random.beta(0.5, 0.5)
        influence_of_others = np.random.beta(0.5, 0.5, size=n_agents).astype(np.float64)
        
        agent = Agent(
            index=index,
            opinion=opinion,
            influence_of_others=influence_of_others
        )
        return agent
            
            