from model import SocialNetwork
from view import View
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from typing import List


class Controller:
    def __init__(self, social_network: SocialNetwork, n_iterations: int) -> None:
        """
        This class serves as the controller for the DeGroot simulation, managing the interaction between the model (SocialNetwork) and the view (View).

        Attributes:
            social_network (SocialNetwork): An instance of the SocialNetwork class representing the network of agents.
            n_iterations (int): The number of iterations to run the simulation.
            opinion_history (List[np.ndarray]): A list to store the opinion vectors at each iteration. This is automatically initialized with the initial opinion vector of the social network.
            influence_matrix_history (deque): A deque to store the influence matrices at each iteration, with a maximum length of 10 for memory efficiency. This is automatically initialized with the initial influence matrix of the social network.
            backward_product (np.ndarray): A matrix that accumulates the product of influence matrices over iterations, used for verifying the final opinion vector, (only in the run_simulation (v1) method). This is automatically initialized as the identity matrix.
            network_graphs (List[plt.Figure]): A list to store the network graphs at each iteration for visualization purposes. This is automatically initialized with the initial network graph of the social network.
            backward_product_history (List[np.ndarray]): A list to store the backward products at each iteration for analysis and verification. This is automatically initialized with the initial backward product (identity matrix).
        """
        self.social_network = social_network
        self.n_iterations = n_iterations
        self.opinion_history: List[np.ndarray] = []
        self.influence_matrix_history: deque = deque(maxlen=10)
        self.backward_product: np.ndarray = None
        self.network_graphs: List[plt.Figure] = []
        self.backward_product_history: List[np.ndarray] = []

        # Initialize backward_product and histories with initial state
        self.append_opinion_vector()
        self.append_influence_matrix()
        self._initial_influence_matrix: np.ndarray = self.influence_matrix_history[0]
        self.append_network_graph()
        self.backward_product = np.eye(self.social_network.n_agents, dtype=np.float64)
    
    def append_opinion_vector(self) -> None:
        self.opinion_history.append(self.social_network.get_opinion_vector().copy())

    def append_influence_matrix(self) -> None:
        self.influence_matrix_history.append(self.social_network.get_influence_matrix().copy())

    def append_network_graph(self) -> None:
        self.network_graphs.append(self.social_network.get_graph().copy())

    def append_backward_product(self) -> None:
        self.backward_product_history.append(self.backward_product.copy())

    def _format_influence_matrix(self, mat: np.ndarray) -> str:
        """
        This method formats the influence matrix for better readability when printing.
        It prints each row of the matrix on a new line
        """
        formatted_rows = []
        for row in mat:
            formatted_row = " ".join(f"{value:.10f}" for value in row)
            formatted_rows.append(formatted_row)
        return "\n".join(formatted_rows)
    
    def run_simulation(self) -> None:
        """
        This method runs the DeGroot simulation for n_iterations.

        At each iteration, it updates the backward product, which is a piece of information that belongs to the Controller, not to the SocialNetwork.

        Then, it updates the opinions and influences in the SocialNetwork, and appends the new states to the histories.

        This method (v1) is useful for recreating results such as those stated in the paper by Chatterjee and Seneta (1977) for "open-minded" agents.
        """
        for iteration in range(self.n_iterations):
            # self.backward_product = self.social_network.get_influence_matrix() @ self.backward_product
            # self.append_backward_product()

            self.social_network.update_opinions()
            self.social_network.update_influences()
            self.social_network.update_graph()

            self.append_opinion_vector()
            self.append_influence_matrix()
            self.append_network_graph()

    def run_simulation_v2(self) -> None:
        """
        This method runs the DeGroot simulation for n_iterations, but it uses the update_influences_v2 method of the SocialNetwork, which updates the influence matrix based on a distribution function and the last opinion vector of all agents in the network (useful for implementing dynamic influence mechanisms driven by homophily).
        
        At each iteration, it updates the backward product, which is a piece of information that belongs to the Controller, not to the SocialNetwork.

        Then, it updates the opinions and influences in the SocialNetwork, and appends the new states to the histories.

        The main difference with the run_simulation (v1) method is that the influence matrix is updated using the distribution function and the last opinion vector, which allows for dynamic weights based on opinion differences between agents, rather than a fixed influence matrix.
        """
        for iteration in range(self.n_iterations):
            # self.backward_product = self.social_network.get_influence_matrix() @ self.backward_product
            # self.append_backward_product()

            self.social_network.update_opinions()
            self.social_network.update_influences_v2(self.opinion_history[-1])
            self.social_network.update_graph()

            self.append_opinion_vector()
            self.append_influence_matrix()
            self.append_network_graph()

    def plot_opinion_history(self, show_labels: bool = True) -> None:
        View.plot_opinion_history(self.opinion_history, self.social_network.n_agents, show_labels=show_labels)
    
    def display_network_graphs_animation(self, include_self_loops: bool = False, filter_vertex: int = None) -> None:
        View.display_network_graphs_animation(self.network_graphs, self.social_network, include_self_loops=include_self_loops, filter_vertex=filter_vertex)     

    def get_first_opinion_vector(self) -> np.ndarray:
        return self.opinion_history[0]
    
    def get_last_opinion_vector(self) -> np.ndarray:
        return self.opinion_history[-1]
    
    def get_opinion_history(self) -> List[np.ndarray]:
        return self.opinion_history
    
    def get_first_influence_matrix(self) -> np.ndarray:
        return self._initial_influence_matrix

    def get_last_influence_matrix(self) -> np.ndarray:
        return self.influence_matrix_history[-1]

    def get_influence_matrix_history(self) -> List[np.ndarray]:
        return list(self.influence_matrix_history)
    
    def get_backward_product_history(self) -> List[np.ndarray]:
        return self.backward_product_history
    
    def get_final_opinion_via_backward_product(self) -> np.ndarray:
        """
        It is possible to compute the final opinion vector by multiplying the initial opinion vector by the backward product.
        This is just an easy and alternative way to verify that the simulation (v1) was run correctly.
        """
        return self.backward_product @ self.get_first_opinion_vector()

    def print_current_network_graph(self, include_self_loops: bool = False) -> None:
        graph = self.network_graphs[-1]
        position = self.social_network.get_node_positions()
        View.display_network_graph(graph, position, include_self_loops)

    def print_opinion_history(self) -> None:
        for i, opinion_vector in enumerate(self.opinion_history):
            print(f"Iteration {i}:")
            print(opinion_vector)
    
    def print_influence_matrix_history(self) -> None:
        for i, influence_matrix in enumerate(self.influence_matrix_history):
            print(f"Iteration {i}:")
            print((influence_matrix))
            print()

    def print_last_influence_matrices(self, n: int = 5) -> None:
        entries = list(self.influence_matrix_history)[-n:]
        last_iter = len(self.opinion_history) - 1
        first_shown_iter = last_iter - len(entries) + 1
        for i, influence_matrix in enumerate(entries):
            print(f"Iteration {first_shown_iter + i}:")
            print(self._format_influence_matrix(influence_matrix))
            print()

    def print_backward_product_history(self) -> None:
        for i, backward_product in enumerate(self.backward_product_history):
            print(f"Iteration {i}:")
            print(backward_product)
