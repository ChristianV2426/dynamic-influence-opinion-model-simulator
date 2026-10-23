from typing import Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import networkx as nx
import numpy as np


class View:
    """Plot recorded simulation states."""

    @staticmethod
    def _opinion_matrix(history: Sequence[np.ndarray]) -> np.ndarray:
        if not history:
            raise ValueError("opinion history must contain at least the initial state")

        values = np.stack(history)
        if values.ndim != 2 or not np.all(np.isfinite(values)):
            raise ValueError("opinion history must contain equal-length, finite vectors")

        return values

    @staticmethod
    def _agent_names(agent_names: Optional[Sequence[str]], n_agents: int) -> Sequence[str]:
        if agent_names is None:
            return [f"Agent {index}" for index in range(n_agents)]

        if len(agent_names) != n_agents:
            raise ValueError("agent_names must contain one name per agent")

        return agent_names

    @staticmethod
    def plot_opinion_history(
        opinion_history: Sequence[np.ndarray],
        agent_names: Optional[Sequence[str]] = None,
        show_labels: bool = True,
    ):
        """Plot each agent's internal opinion from round zero onward."""
        opinions = View._opinion_matrix(opinion_history)
        names = View._agent_names(agent_names, opinions.shape[1])
        rounds = np.arange(opinions.shape[0])
        colors = plt.get_cmap("tab20")

        fig, ax = plt.subplots()
        for index, name in enumerate(names):
            ax.plot(rounds, opinions[:, index], color=colors(index % 20), label=name)

        ax.set(xlabel="Round", ylabel="Internal/Private opinion", title="Opinion evolution", ylim=(-0.02, 1.02))
        ax.set_xlim(0, max(1, len(rounds) - 1))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True)
        if show_labels:
            ax.legend()

        plt.show()
        return fig

    @staticmethod
    def plot_private_public_opinion_history(
        opinion_history: Sequence[np.ndarray],
        public_opinion_history: Sequence[np.ndarray],
        speaking_history: Sequence[np.ndarray],
        agent_names: Optional[Sequence[str]] = None,
        show_labels: bool = True,
    ):
        """Use one color per agent, solid internal/private lines, dashed public lines, and squares for rounds where the agent is silent."""
        opinions = View._opinion_matrix(opinion_history)
        public_opinions = View._opinion_matrix(public_opinion_history)
        speaking = np.stack(speaking_history)

        if public_opinions.shape != opinions.shape or speaking.shape != opinions.shape:
            raise ValueError("opinion, public opinion, and speaking histories must have matching shapes")

        names = View._agent_names(agent_names, opinions.shape[1])
        rounds = np.arange(opinions.shape[0])
        colors = plt.get_cmap("tab20")
        silence_labeled = False

        fig, ax = plt.subplots(figsize=(9, 5))
        for index, name in enumerate(names):
            color = colors(index % 20)
            ax.plot(rounds, public_opinions[:, index], "--", color=color, label=f"{name} public")
            ax.plot(rounds, opinions[:, index], "-", color=color, label=f"{name} internal/private")

            silent = ~speaking[:, index].astype(np.bool_)
            if np.any(silent):
                ax.scatter(
                    rounds[silent], public_opinions[silent, index], marker="s", color=color,
                    edgecolors="black", zorder=3,
                    label="Silent (speaking=False)" if not silence_labeled else None,
                )
                silence_labeled = True

        ax.set(xlabel="Round", ylabel="Opinion", title="Internal and public opinions", ylim=(-0.02, 1.02))
        ax.set_xlim(0, max(1, len(rounds) - 1))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True)
        if show_labels:
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
        fig.tight_layout()

        plt.show()
        return fig

    @staticmethod
    def plot_minimum_influence_history(
        minimum_influence_history: Sequence[float],
        use_natural_log: bool = True,
    ):
        """Plot the minimum positive influence by round, using its natural logarithm by default."""
        if not minimum_influence_history:
            raise ValueError("minimum influence history is empty; enable record_minimum_influence before running")

        values = np.asarray(minimum_influence_history, dtype=np.float64)
        if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("minimum influence history must contain finite, positive values")

        fig, ax = plt.subplots()
        ax.plot(np.arange(values.size), np.log(values) if use_natural_log else values)
        ax.set(
            xlabel="Round",
            ylabel="ln(Minimum positive influence)" if use_natural_log else "Minimum positive influence",
            title="Minimum influence evolution",
        )
        ax.set_xlim(0, max(1, values.size - 1))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True)

        plt.show()
        return fig

    @staticmethod
    def display_network_graphs_animation(
        influence_matrix_history: Sequence[np.ndarray],
        agent_names: Optional[Sequence[str]] = None,
        include_self_loops: bool = False,
        filter_vertex: Optional[int] = None,
    ):
        """Navigate influence matrices with arrow keys, for showing influence evolution in the network."""
        if not influence_matrix_history:
            raise ValueError("influence matrix history is empty; enable record_influence_history before running")

        initial_matrix = np.asarray(influence_matrix_history[0], dtype=np.float64)
        if initial_matrix.ndim != 2 or initial_matrix.shape[0] != initial_matrix.shape[1]:
            raise ValueError("influence matrices must be square")

        n_agents = initial_matrix.shape[0]
        names = View._agent_names(agent_names, n_agents)
        if filter_vertex is not None and not 0 <= filter_vertex < n_agents:
            raise ValueError("filter_vertex must be an agent index")

        initial_graph = nx.from_numpy_array(initial_matrix.T, create_using=nx.DiGraph)
        positions = nx.spring_layout(initial_graph, seed=42, weight=None)
        fig, ax = plt.subplots()
        current_round = [0]

        def draw_round(index: int) -> None:
            matrix = np.asarray(influence_matrix_history[index], dtype=np.float64)
            if matrix.shape != initial_matrix.shape or not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
                raise ValueError("influence matrices must have matching shapes and non-negative, finite weights")

            graph = nx.from_numpy_array(matrix.T, create_using=nx.DiGraph)
            edges = [
                edge for edge in graph.edges
                if (include_self_loops or edge[0] != edge[1])
                and (filter_vertex is None or edge[1] == filter_vertex)
            ]
            curved = [edge for edge in edges if edge[0] != edge[1] and (edge[1], edge[0]) in edges]
            straight = [edge for edge in edges if edge not in curved]

            ax.clear()
            nx.draw_networkx_nodes(graph, positions, ax=ax)
            nx.draw_networkx_labels(graph, positions, labels=dict(enumerate(names)), ax=ax)

            for selected, connection_style in ((curved, "arc3,rad=0.15"), (straight, "arc3")):
                if not selected:
                    continue

                weights = [graph[u][v]["weight"] for u, v in selected]
                nx.draw_networkx_edges(
                    graph, positions, edgelist=selected, width=[1 + 4 * weight for weight in weights],
                    connectionstyle=connection_style, arrows=True, ax=ax,
                )
                nx.draw_networkx_edge_labels(
                    graph, positions,
                    edge_labels={(u, v): f"{graph[u][v]['weight']:.3g}" for u, v in selected},
                    connectionstyle=connection_style, ax=ax,
                )

            ax.set_title(f"Influence network - round {index}")
            ax.set_axis_off()
            fig.canvas.draw_idle()

        def on_key(event) -> None:
            if event.key == "right":
                current_round[0] = (current_round[0] + 1) % len(influence_matrix_history)
            elif event.key == "left":
                current_round[0] = (current_round[0] - 1) % len(influence_matrix_history)
            else:
                return

            draw_round(current_round[0])

        fig.canvas.mpl_connect("key_press_event", on_key)
        draw_round(0)
        plt.show()
        return fig
