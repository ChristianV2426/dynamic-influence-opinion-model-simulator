import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from controller.simulation_parser import SimulationParser
from view.view import View


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a dynamic influence opinion simulation.")

    parser.add_argument("configuration", type=Path, help="Path to the simulation JSON file")

    parser.add_argument(
        "--plot", nargs="+", choices=("opinions", "opinions-with-silence", "private-public", "network", "minimum-influence"),
        default=["opinions"], help="Plots to display (default: opinions)",
    )
    parser.add_argument(
        "--agent-legend", action=argparse.BooleanOptionalAction, default=True,
        help=f"Show the agent-name/color legend when there are at most {View.MAX_AGENT_LEGEND_ENTRIES} agents (default: on)",
    )
    args = parser.parse_args()

    try:
        controller = SimulationParser.from_file(args.configuration)

        if "minimum-influence" in args.plot and not controller.record_minimum_influence:
            raise ValueError("minimum-influence plot requires recordMinimumInfluence: true in the JSON")

        initial_matrix = controller.social_network.influence_matrix if "network" in args.plot else None

        controller.run_simulation()

        names = [agent.name for agent in controller.social_network.agents]

        for plot in args.plot:
            if plot == "opinions":
                View.plot_opinion_history(
                    controller.opinion_history, names, show_labels=args.agent_legend, show=False)

            elif plot == "opinions-with-silence":
                View.plot_opinion_history_with_silence(
                    controller.opinion_history, controller.speaking_history, names,
                    show_labels=args.agent_legend, show=False)

            elif plot == "private-public":
                View.plot_private_public_opinion_history(
                    controller.opinion_history, controller.public_opinion_history,
                    controller.speaking_history, names, show_labels=args.agent_legend, show=False)

            elif plot == "network":
                history = controller.influence_matrix_history or [initial_matrix]
                View.display_network_graphs_animation(
                    history, names, include_self_loops=True, show=False)

            else:
                View.plot_minimum_influence_history(
                    controller.minimum_influence_history, show=False)

    except (OSError, ValueError) as exc:
        print(f"Simulation error: {exc}", file=sys.stderr)
        return 1

    print(f"Completed rounds: {controller.completed_iterations}")
    print(f"Stop reason: {controller.stop_reason}")
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
