import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from model.social_network import SocialNetwork


class View:
    """
    This is a class that handles the visualization of the simulations and its derived data. It is designed to work with the Controller and SocialNetwork classes providing, among others, methods to plot opinion histories and display network graphs.
    """

    @staticmethod
    def plot_opinion_history(opinion_history: List[np.ndarray], n_agents: int, show_labels: bool = True) -> None:
        """
        This method plots the opinion history of all agents over time.

        Attributes:
            opinion_history (List[np.ndarray]): A list of numpy arrays, where each array represents the opinion vector of all agents at a specific time step.
            n_agents (int): The total number of agents in the network.
            show_labels (bool): A flag to indicate whether to show labels for each agent in the plot. Default is True.
        """
        # Reshape the opinion history into a 2D array where each row corresponds to an agent and each column corresponds to a time step
        opinions = np.hstack(opinion_history)
        n_iterations = len(opinion_history)
        iteration_range = range(n_iterations)

        plt.figure()
        for agent_index in range(n_agents):
            plt.plot(
                iteration_range,
                opinions[agent_index, :],
                label=f"Agent {agent_index}"
            )

        plt.xlim(0, n_iterations - 1)
        plt.ylim(0, 1)
        plt.xlabel("Time")
        plt.ylabel("Opinion")
        plt.title("Opinion Evolution Over Time")
        if show_labels:
            plt.legend(loc='best')
        plt.grid(True)

        plt.show()

    @staticmethod
    def display_network_graph(graph: nx.DiGraph, position: dict, include_self_loops: bool = False, filter_vertex: int = None, ax=None) -> None:
        """
        This method prints the network graph using Matplotlib and NetworkX.
        It is necessary to make sure that when two agents influence each other, the edges are drawn with curves to avoid overlapping.
        In the case of one-directional influence, straight edges are drawn.
        It is also possible to exclude self-loops from the visualization by setting include_self_loops to False.
        If filter_vertex is provided, only edges involving that vertex are drawn.

        Attributes:
            graph (nx.DiGraph): A directed graph representing the influence relationships between agents.
            position (dict): A dictionary mapping agent indices to their positions in the graph layout for visualization purposes.
            include_self_loops (bool): A flag to indicate whether to include self-loops in the visualization. Default is False.
            filter_vertex (int): An optional agent index to filter the edges displayed in the graph. If provided, only edges involving this vertex will be shown.
            ax: An optional Matplotlib Axes object to draw on. If not provided, a new figure and axes will be created.
        """
        if ax is None:
            fig, ax = plt.subplots()

        # Get edge weights for labeling
        edge_labels = {edge: round(weight, 4) for edge, weight in nx.get_edge_attributes(graph, 'weight').items()}

        # Get edges to draw
        if include_self_loops:
            edges_to_draw = graph.edges()
        else:
            # Exclude self-loops
            self_loops = set(nx.selfloop_edges(graph))
            edges_to_draw = [
                edge for edge in graph.edges() if edge not in self_loops]

        # Filter edges to only include those directed toward the filter_vertex
        if filter_vertex is not None:
            edges_to_draw = [
                edge for edge in edges_to_draw
                if edge[1] == filter_vertex
            ]
            edge_labels = {
                edge: label for edge, label in edge_labels.items()
                if edge[1] == filter_vertex
            }

        # Draw nodes and their labels
        nx.draw_networkx_nodes(graph, position, ax=ax)
        nx.draw_networkx_labels(graph, position, ax=ax, font_size=10)

        # Draw curved edges and their labels
        curved_edges = [edge for edge in edges_to_draw if (
            edge[1], edge[0]) in edges_to_draw]
        curved_edges_labels = {edge: label for edge,
                               label in edge_labels.items() if edge in curved_edges}
        nx.draw_networkx_edges(
            G=graph,
            pos=position,
            edgelist=curved_edges,
            connectionstyle='arc3, rad=0.15',
            ax=ax
        )
        nx.draw_networkx_edge_labels(
            G=graph,
            pos=position,
            edge_labels=curved_edges_labels,
            connectionstyle='arc3, rad=0.15',
            font_size=8,
            label_pos=0.5,
            ax=ax
        )

        # Draw straight edges and their labels
        straight_edges = list(set(edges_to_draw) - set(curved_edges))
        straight_edges_labels = {
            edge: label for edge, label in edge_labels.items() if edge in straight_edges}
        nx.draw_networkx_edges(
            G=graph,
            pos=position,
            edgelist=straight_edges,
            ax=ax
        )
        nx.draw_networkx_edge_labels(
            G=graph,
            pos=position,
            edge_labels=straight_edges_labels,
            font_size=8,
            label_pos=0.5,
            ax=ax
        )

    @staticmethod
    def display_network_graphs_animation(network_graphs: List[nx.DiGraph], social_network: SocialNetwork, include_self_loops: bool = False, filter_vertex: int = None) -> None:
        """
        This method displays a Matplotlib figure that allows navigating through the network graphs of each iteration using left and right arrow keys.

        Attributes:
            network_graphs (List[nx.DiGraph]): A list of directed graphs representing the influence relationships between agents at each iteration.
            social_network (SocialNetwork): The SocialNetwork object containing the agents and their relationships.
            include_self_loops (bool): A flag to indicate whether to include self-loops in the visualization. Default is False.
            filter_vertex (int): An optional agent index to filter the edges displayed in the graph. If provided, only edges involving this vertex will be shown.
        """
        fig, ax = plt.subplots()
        position = social_network.get_node_positions()
        total_graphs = len(network_graphs)
        # Mutable container to track the current graph index
        current_index = [0]

        def draw_graph(index: int) -> None:
            ax.clear()
            View.display_network_graph(
                graph=network_graphs[index],
                position=position,
                include_self_loops=include_self_loops,
                filter_vertex=filter_vertex,
                ax=ax
            )
            ax.set_title(f"Iteration {index}")
            fig.canvas.draw_idle()

        def on_key(event) -> None:
            if event.key == 'right':
                current_index[0] = (current_index[0] + 1) % total_graphs
            elif event.key == 'left':
                current_index[0] = (current_index[0] - 1) % total_graphs

            draw_graph(current_index[0])

        fig.canvas.mpl_connect('key_press_event', on_key)
        draw_graph(current_index[0])
        plt.show()
