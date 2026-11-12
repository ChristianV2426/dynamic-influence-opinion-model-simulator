from typing import Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import networkx as nx
import numpy as np

class View:
    """Plot recorded simulation states."""

    OPINION_COLOR_PALETTE = "tab10"
    OPINION_COLOR_PALETTE_LARGE = "tab20"
    MAX_AGENT_LEGEND_ENTRIES = 20
    OPINION_Y_MARGIN = 0.025
    OPINION_X_MARGIN = 0.01
    AXIS_LABEL_FONT_SIZE = 16
    TICK_LABEL_FONT_SIZE = 16
    LEGEND_FONT_SIZE = 14
    LEGEND_TITLE_FONT_SIZE = 16
    FIGURE_SIZE = (9.4, 6)
    TOLERANCE_BAND_ALPHA = 0.18

    @staticmethod
    def _opinion_matrix(history: Sequence[np.ndarray]) -> np.ndarray:
        if not history:
            raise ValueError(
                "opinion history must contain at least the initial state")

        values = np.stack(history)
        if values.ndim != 2 or not np.all(np.isfinite(values)):
            raise ValueError(
                "opinion history must contain equal-length, finite vectors")

        return values

    @staticmethod
    def _agent_names(agent_names: Optional[Sequence[str]], n_agents: int) -> Sequence[str]:
        if agent_names is None:
            return [f"Agent {index}" for index in range(n_agents)]

        if len(agent_names) != n_agents:
            raise ValueError("agent_names must contain one name per agent")

        return agent_names

    @staticmethod
    def _opinion_colors(n_agents: int):
        palette = (View.OPINION_COLOR_PALETTE if n_agents <= plt.get_cmap(View.OPINION_COLOR_PALETTE).N
                   else View.OPINION_COLOR_PALETTE_LARGE)
        return plt.get_cmap(palette)

    @staticmethod
    def _tolerance_radii(
        tolerance_radii: Optional[Sequence[Optional[float]]],
        n_agents: int,
    ) -> Sequence[Optional[float]]:
        """Validate one optional tolerance radius per agent."""
        if tolerance_radii is None:
            return [None] * n_agents

        if len(tolerance_radii) != n_agents:
            raise ValueError("tolerance_radii must contain one value per agent")

        radii = []
        for radius in tolerance_radii:
            if radius is None:
                radii.append(None)
                continue

            if isinstance(radius, (bool, np.bool_)):
                raise ValueError("tolerance radii must be numbers in [0, 1] or None")

            try:
                value = float(radius)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("tolerance radii must be numbers in [0, 1] or None") from exc

            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("tolerance radii must be numbers in [0, 1] or None")

            radii.append(value)

        return radii

    @staticmethod
    def _plot_tolerance_bands(
        ax,
        rounds: np.ndarray,
        opinions: np.ndarray,
        tolerance_radii: Sequence[Optional[float]],
        colors,
    ) -> None:
        """Shade the tolerance interval around each agent's internal opinion."""
        for index, radius in enumerate(tolerance_radii):
            if radius is None or radius == 0:
                continue

            color = colors(index % colors.N)
            ax.fill_between(
                rounds,
                np.clip(opinions[:, index] - radius, 0, 1),
                np.clip(opinions[:, index] + radius, 0, 1),
                color=color,
                alpha=View.TOLERANCE_BAND_ALPHA,
                linewidth=0,
                zorder=0,
            )

    @staticmethod
    def _configure_time_axis(ax, last_time: int) -> None:
        """Use five equal time intervals when the final time is divisible by five."""
        if last_time > 0 and last_time % 5 == 0:
            ax.set_xticks(np.linspace(0, last_time, 6, dtype=int))
        else:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

    @staticmethod
    def _add_side_legends(fig, sections) -> None:
        """Place titled legend sections consecutively in the right-side margin."""
        fig.subplots_adjust(right=0.72)
        x_position = 0.75
        y_position = 0.93

        for title, handles in sections:
            if not handles:
                continue

            legend = fig.legend(
                handles=handles,
                loc="upper left",
                bbox_to_anchor=(x_position, y_position),
                bbox_transform=fig.transFigure,
                ncol=1,
                fontsize=View.LEGEND_FONT_SIZE,
                title=title,
                title_fontproperties={
                    "weight": "bold",
                    "size": View.LEGEND_TITLE_FONT_SIZE,
                },
            )
            fig.canvas.draw()
            bounds = legend.get_window_extent(fig.canvas.get_renderer())
            y_position = fig.transFigure.inverted().transform(
                (bounds.x0, bounds.y0)
            )[1] - 0.02

    @staticmethod
    def plot_opinion_history(
        opinion_history: Sequence[np.ndarray],
        agent_names: Optional[Sequence[str]] = None,
        show_labels: bool = True,
        show: bool = True,
    ):
        """Plot each agent's opinion from round zero onward."""
        opinions = View._opinion_matrix(opinion_history)
        names = View._agent_names(agent_names, opinions.shape[1])
        rounds = np.arange(opinions.shape[0])
        colors = View._opinion_colors(len(names))

        fig, ax = plt.subplots(figsize=View.FIGURE_SIZE)
        for index, name in enumerate(names):
            ax.plot(rounds, opinions[:, index], color=colors(
                index % colors.N), label=name)

        ax.set_ylim(-View.OPINION_Y_MARGIN, 1 + View.OPINION_Y_MARGIN)
        ax.set_xlabel("Time", fontweight="bold", fontsize=View.AXIS_LABEL_FONT_SIZE)
        ax.set_ylabel("Opinion", fontweight="bold", fontsize=View.AXIS_LABEL_FONT_SIZE)
        ax.tick_params(axis="both", labelsize=View.TICK_LABEL_FONT_SIZE)
        ax.set_yticks(np.linspace(0, 1, 11))
        last_round = max(1, len(rounds) - 1)
        margin = View.OPINION_X_MARGIN * last_round
        ax.set_xlim(-margin, last_round + margin)
        View._configure_time_axis(ax, last_round)
        if show_labels and len(names) <= View.MAX_AGENT_LEGEND_ENTRIES:
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1),
                      ncol=1, fontsize=View.LEGEND_FONT_SIZE, title="Agents",
                      title_fontproperties={"weight": "bold", "size": View.LEGEND_TITLE_FONT_SIZE})
            fig.tight_layout()

        if show:
            plt.show()
        return fig

    @staticmethod
    def plot_opinion_history_with_silence(
        opinion_history: Sequence[np.ndarray],
        speaking_history: Sequence[np.ndarray],
        agent_names: Optional[Sequence[str]] = None,
        show_labels: bool = True,
        show: bool = True,
        tolerance_radii: Optional[Sequence[Optional[float]]] = None,
        show_tolerance_bands: bool = False,
    ):
        """Plot opinions with silence markers and, optionally, tolerance bands."""
        opinions = View._opinion_matrix(opinion_history)
        speaking = np.stack(speaking_history)
        if speaking.shape != opinions.shape:
            raise ValueError("speaking history must match opinion history")

        if not isinstance(show_tolerance_bands, (bool, np.bool_)):
            raise ValueError("show_tolerance_bands must be a boolean")

        names = View._agent_names(agent_names, opinions.shape[1])
        radii = View._tolerance_radii(tolerance_radii, len(names))
        fig = View.plot_opinion_history(opinion_history, names, show_labels=False, show=False)
        ax = fig.axes[0]
        rounds = np.arange(opinions.shape[0])
        colors = View._opinion_colors(len(names))

        if show_tolerance_bands:
            View._plot_tolerance_bands(ax, rounds, opinions, radii, colors)

        for index in range(len(names)):
            color = colors(index % colors.N)
            speaking_now = speaking[:, index].astype(np.bool_)
            if np.any(speaking_now):
                ax.scatter(rounds[speaking_now], opinions[speaking_now, index],
                           marker="o", color=color, edgecolors="black", zorder=3)

            silent = ~speaking_now
            if np.any(silent):
                ax.scatter(rounds[silent], opinions[silent, index],
                           marker="s", color=color, edgecolors="black", zorder=3)

        agent_handles = ([Line2D([0], [0], color=colors(index % colors.N), label=name)
                          for index, name in enumerate(names)]
                         if show_labels and len(names) <= View.MAX_AGENT_LEGEND_ENTRIES else [])
        status_handles = [
            Line2D([0], [0], color="black", marker="o", linestyle="None", label="Non-silent"),
            Line2D([0], [0], color="black", marker="s", linestyle="None", label="Silent"),
        ]
        View._add_side_legends(
            fig,
            [("Agents", agent_handles), ("Silent status", status_handles)],
        )

        if show:
            plt.show()
        return fig

    @staticmethod
    def plot_private_public_opinion_history(
        opinion_history: Sequence[np.ndarray],
        public_opinion_history: Sequence[np.ndarray],
        speaking_history: Sequence[np.ndarray],
        agent_names: Optional[Sequence[str]] = None,
        show_labels: bool = True,
        show: bool = True,
        tolerance_radii: Optional[Sequence[Optional[float]]] = None,
        show_tolerance_bands: bool = False,
    ):
        """Plot opinions and public opinions and, optionally, tolerance bands."""
        opinions = View._opinion_matrix(opinion_history)
        public_opinions = View._opinion_matrix(public_opinion_history)
        speaking = np.stack(speaking_history)

        if public_opinions.shape != opinions.shape or speaking.shape != opinions.shape:
            raise ValueError(
                "opinion, public opinion, and speaking histories must have matching shapes")

        if not isinstance(show_tolerance_bands, (bool, np.bool_)):
            raise ValueError("show_tolerance_bands must be a boolean")

        names = View._agent_names(agent_names, opinions.shape[1])
        radii = View._tolerance_radii(tolerance_radii, len(names))
        rounds = np.arange(opinions.shape[0])
        colors = View._opinion_colors(len(names))

        fig, ax = plt.subplots(figsize=View.FIGURE_SIZE)
        if show_tolerance_bands:
            View._plot_tolerance_bands(ax, rounds, opinions, radii, colors)

        for index in range(len(names)):
            color = colors(index % colors.N)
            ax.plot(rounds, public_opinions[:, index],
                    "--", color=color)
            ax.plot(rounds, opinions[:, index], "-",
                    color=color)

            speaking_now = speaking[:, index].astype(np.bool_)
            if np.any(speaking_now):
                ax.scatter(
                    rounds[speaking_now], opinions[speaking_now, index],
                    marker="o", color=color, edgecolors="black", zorder=3,
                )

            silent = ~speaking_now
            if np.any(silent):
                ax.scatter(
                    rounds[silent], opinions[silent, index], marker="s", color=color,
                    edgecolors="black", zorder=3,
                )

        ax.set(ylim=(-View.OPINION_Y_MARGIN, 1 + View.OPINION_Y_MARGIN))
        ax.set_xlabel("Time", fontweight="bold", fontsize=View.AXIS_LABEL_FONT_SIZE)
        ax.set_ylabel("Opinion", fontweight="bold", fontsize=View.AXIS_LABEL_FONT_SIZE)
        ax.tick_params(axis="both", labelsize=View.TICK_LABEL_FONT_SIZE)
        ax.set_yticks(np.linspace(0, 1, 11))
        last_round = max(1, len(rounds) - 1)
        margin = View.OPINION_X_MARGIN * last_round
        ax.set_xlim(-margin, last_round + margin)
        View._configure_time_axis(ax, last_round)
        agent_handles = ([Line2D([0], [0], color=colors(index % colors.N), label=name)
                          for index, name in enumerate(names)]
                         if show_labels and len(names) <= View.MAX_AGENT_LEGEND_ENTRIES else [])
        plot_handles = [
            Line2D([0], [0], color="black", linestyle="-", label="Opinion"),
            Line2D([0], [0], color="black", linestyle="--", label="Public opinion"),
        ]
        status_handles = [
            Line2D([0], [0], color="black", marker="o", linestyle="None", label="Non-silent"),
            Line2D([0], [0], color="black", marker="s", linestyle="None", label="Silent"),
        ]
        View._add_side_legends(
            fig,
            [
                ("Agents", agent_handles),
                ("Silent status", status_handles),
                ("Plot", plot_handles),
            ],
        )

        if show:
            plt.show()
        return fig

    @staticmethod
    def plot_minimum_influence_history(
        minimum_influence_history: Sequence[float],
        show: bool = True,
    ):
        """Plot the base-10 logarithm of the minimum positive influence by time."""
        if not minimum_influence_history:
            raise ValueError(
                "minimum influence history is empty; enable record_minimum_influence before running")

        values = np.asarray(minimum_influence_history, dtype=np.float64)
        if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(
                "minimum influence history must contain finite, positive values")

        fig, ax = plt.subplots(figsize=View.FIGURE_SIZE)
        ax.plot(np.arange(values.size), np.log10(values))
        ax.set_xlabel("Time", fontweight="bold", fontsize=View.AXIS_LABEL_FONT_SIZE)
        ax.set_ylabel(
            "log₁₀(Minimum influence)",
            fontweight="bold",
            fontsize=View.AXIS_LABEL_FONT_SIZE,
        )
        ax.tick_params(axis="both", labelsize=View.TICK_LABEL_FONT_SIZE)
        last_time = max(1, values.size - 1)
        margin = View.OPINION_X_MARGIN * last_time
        ax.set_xlim(-margin, last_time + margin)
        View._configure_time_axis(ax, last_time)

        if show:
            plt.show()
        return fig

    @staticmethod
    def display_network_graphs_animation(
        influence_matrix_history: Sequence[np.ndarray],
        agent_names: Optional[Sequence[str]] = None,
        include_self_loops: bool = False,
        filter_vertex: Optional[int] = None,
        show: bool = True,
    ):
        """Navigate influence matrices with arrow keys, for showing influence evolution in the network."""
        if not influence_matrix_history:
            raise ValueError(
                "influence matrix history is empty; enable record_influence_history before running")

        initial_matrix = np.asarray(
            influence_matrix_history[0], dtype=np.float64)
        if initial_matrix.ndim != 2 or initial_matrix.shape[0] != initial_matrix.shape[1]:
            raise ValueError("influence matrices must be square")

        n_agents = initial_matrix.shape[0]
        names = View._agent_names(agent_names, n_agents)
        if filter_vertex is not None and not 0 <= filter_vertex < n_agents:
            raise ValueError("filter_vertex must be an agent index")

        initial_graph = nx.from_numpy_array(
            initial_matrix.T, create_using=nx.DiGraph)
        positions = nx.spring_layout(initial_graph, seed=42, weight=None)
        fig, ax = plt.subplots(figsize=View.FIGURE_SIZE)
        current_round = [0]

        def draw_round(index: int) -> None:
            matrix = np.asarray(
                influence_matrix_history[index], dtype=np.float64)
            if matrix.shape != initial_matrix.shape or not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
                raise ValueError(
                    "influence matrices must have matching shapes and non-negative, finite weights")

            graph = nx.from_numpy_array(matrix.T, create_using=nx.DiGraph)
            edges = [
                edge for edge in graph.edges
                if (include_self_loops or edge[0] != edge[1])
                and (filter_vertex is None or edge[1] == filter_vertex)
            ]
            curved = [edge for edge in edges if edge[0] !=
                      edge[1] and (edge[1], edge[0]) in edges]
            straight = [edge for edge in edges if edge not in curved]

            ax.clear()
            nx.draw_networkx_nodes(graph, positions, ax=ax)
            nx.draw_networkx_labels(
                graph, positions, labels=dict(enumerate(names)), ax=ax)

            for selected, connection_style in ((curved, "arc3,rad=0.15"), (straight, "arc3")):
                if not selected:
                    continue

                nx.draw_networkx_edges(
                    graph, positions, edgelist=selected, width=1.0,
                    connectionstyle=connection_style, arrows=True, ax=ax,
                )
                nx.draw_networkx_edge_labels(
                    graph, positions,
                    edge_labels={
                        (u, v): f"{graph[u][v]['weight']:.5g}" for u, v in selected},
                    connectionstyle=connection_style, ax=ax,
                )

            ax.set_title(f"Iteration {index}", fontsize=View.AXIS_LABEL_FONT_SIZE)
            ax.set_axis_on()
            ax.set_xticks([])
            ax.set_yticks([])
            fig.canvas.draw_idle()

        def on_key(event) -> None:
            if event.key == "right":
                current_round[0] = (current_round[0] +
                                    1) % len(influence_matrix_history)
            elif event.key == "left":
                current_round[0] = (current_round[0] -
                                    1) % len(influence_matrix_history)
            else:
                return

            draw_round(current_round[0])

        fig.canvas.mpl_connect("key_press_event", on_key)
        draw_round(0)
        if show:
            plt.show()
        return fig
