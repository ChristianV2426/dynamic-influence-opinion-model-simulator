from typing import Callable, Optional, Sequence
import networkx as nx
import numpy as np

from .agent import Agent


class SocialNetwork:
    """
    Represent the agents and their fixed influence connections.

    - agents: Network agents, indexed consecutively from zero; each influence row must contain one entry per agent.
    - distribution_function: One network-wide function that maps the distance between a neighbor's public opinion and the agent's private opinion to a positive influence value; required if any agent has dynamic influence.
    """

    def __init__(
        self,
        agents: Sequence[Agent],
        distribution_function: Optional[Callable[[float], float]] = None,
    ) -> None:
        agent_list = list(agents)

        if not agent_list or any(not isinstance(agent, Agent) for agent in agent_list):
            raise ValueError("agents must be a non-empty sequence of Agent instances")

        self.agents = sorted(agent_list, key=lambda agent: agent.index)
        self.n_agents = len(self.agents)

        if [agent.index for agent in self.agents] != list(range(self.n_agents)):
            raise ValueError("agent indices must be consecutive from zero")

        if len({agent.name for agent in self.agents}) != self.n_agents:
            raise ValueError("agent names must be unique")

        if any(agent.influence_of_others.size != self.n_agents for agent in self.agents):
            raise ValueError("each influence row must have one entry per agent")

        if distribution_function is not None and not callable(distribution_function):
            raise ValueError("distribution_function must be callable")

        if any(agent.dynamic_influence for agent in self.agents) and distribution_function is None:
            raise ValueError("dynamic agents require a distribution_function")

        self.distribution_function = distribution_function
        self._connections = self.influence_matrix > 0

    @classmethod
    def generate_random_network(
        cls,
        n_agents: int,
        neighbors_per_agent: int,
        rewiring_probability: float,
        seed: Optional[int] = None,
        allowed_agent_types: Optional[Sequence[tuple[str, bool]]] = None,
        opinion_alpha: float = 0.5,
        opinion_beta: float = 0.5,
        weight_alpha: float = 0.5,
        weight_beta: float = 0.5,
        tolerance_radius: Optional[float] = None,
        majority_threshold: Optional[float] = None,
        distribution_function: Optional[Callable[[float], float]] = None,
    ) -> "SocialNetwork":
        """
        Generate initial agent states on a connected Watts-Strogatz topology.
        Connectivity is guaranteed, but small-world properties are not checked or guaranteed.
        With at least four initial neighbors, some rewiring can produce a small-world
        graph by retaining local clustering and shortening paths. This is not guaranteed.

        - n_agents: Number of agents and rows in the influence matrix.
        - neighbors_per_agent: Even number of nearby neighbors in the initial ring, before rewiring.
        - rewiring_probability: Chance of rewiring each ring edge, potentially creating shortcuts, in [0, 1].
        - seed: Optional seed that reproduces topology, agent types, opinions, and weights.
        - allowed_agent_types: Distinct (silence_mode, dynamic_influence) pairs; each agent independently chooses one pair with equal probability, so two pairs give each agent a 50/50 choice.
        - opinion_alpha, opinion_beta: Shape of the beta distribution for initial opinions.
        - weight_alpha, weight_beta: Shape of the beta distribution for raw influence weights, before each row is normalized.
        - tolerance_radius, majority_threshold: Shared silence parameters, required if any allowed type uses silence.
        - distribution_function: One network-wide influence update rule, required if any allowed type is dynamic.

        For the parameter allowed_agent_types, the first element of each pair is the agent's silence mode, and the second element is a boolean indicating whether the agent updates its influence weights dynamically between rounds.
        The silence_mode values are "none", "memoryless", and "memory".
        The boolean dynamic_influence says whether that agent updates its influence weights between rounds.
        Possible pairs are:
        - ("none", False): Fixed-influence DeGroot without silence. This is the default option when allowed_agent_types is omitted.
        - ("none", True): Dynamic influence without silence.
        - ("memoryless", False): Memoryless silence with fixed influence.
        - ("memoryless", True): Memoryless silence with dynamic influence.
        - ("memory", False): Silence with memory and fixed influence.
        - ("memory", True): Silence with memory and dynamic influence.

        Every undirected graph edge allows influence in both directions, but the
        two receiving-agent weights are sampled independently. Each agent also
        has self-influence. Beta(0.5, 0.5), the default for both draws, favors
        values near 0 and 1. Since weights are normalized afterward, when the Agent is instantiated,
        the final weights do not necessarily follow the raw beta distribution anymore.
        """
        if isinstance(n_agents, bool) or not isinstance(n_agents, int) or n_agents < 3:
            raise ValueError("n_agents must be an integer of at least 3")

        if (isinstance(neighbors_per_agent, bool) or not isinstance(neighbors_per_agent, int)
                or neighbors_per_agent < 2 or neighbors_per_agent >= n_agents
                or neighbors_per_agent % 2 != 0):
            raise ValueError("neighbors_per_agent must be an even integer in [2, n_agents)")

        rewiring_probability = Agent._unit_interval(rewiring_probability, "rewiring_probability")

        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
            raise ValueError("seed must be a non-negative integer")

        opinion_alpha = Agent._positive_shape_parameter(opinion_alpha, "opinion_alpha")
        opinion_beta = Agent._positive_shape_parameter(opinion_beta, "opinion_beta")
        weight_alpha = Agent._positive_shape_parameter(weight_alpha, "weight_alpha")
        weight_beta = Agent._positive_shape_parameter(weight_beta, "weight_beta")

        # With no type list, every agent is DeGroot. Otherwise sample uniformly from the given pairs.
        agent_types = [("none", False)] if allowed_agent_types is None else list(allowed_agent_types)

        if not agent_types or any(
            not isinstance(agent_type, (tuple, list)) or len(agent_type) != 2
            or agent_type[0] not in Agent.SILENCE_MODES or not isinstance(agent_type[1], bool)
            for agent_type in agent_types
        ):
            raise ValueError("allowed_agent_types must contain (silence_mode, dynamic_influence) pairs")

        if len({tuple(agent_type) for agent_type in agent_types}) != len(agent_types):
            raise ValueError("allowed_agent_types cannot contain duplicates")

        if any(silence_mode != "none" for silence_mode, _ in agent_types):
            if tolerance_radius is None or majority_threshold is None:
                raise ValueError("silence agent types require tolerance_radius and majority_threshold")
            Agent._unit_interval(tolerance_radius, "tolerance_radius")
            Agent._unit_interval(majority_threshold, "majority_threshold")

        if any(dynamic for _, dynamic in agent_types) and distribution_function is None:
            raise ValueError("dynamic agent types require a distribution_function")

        if distribution_function is not None and not callable(distribution_function):
            raise ValueError("distribution_function must be callable")

        # First choose the fixed, connected topology. It determines where weights may be nonzero.
        graph = nx.connected_watts_strogatz_graph(
            n_agents, neighbors_per_agent, rewiring_probability, seed=seed,
        )

        # Give agent types, opinions, and weights their own random generators.
        type_seed, opinion_seed, weight_seed = np.random.SeedSequence(seed).spawn(3)
        type_generator = np.random.default_rng(type_seed)
        opinion_generator = np.random.default_rng(opinion_seed)
        weight_generator = np.random.default_rng(weight_seed)
        agents = []

        for index in range(n_agents):
            # Choose this agent's behavior and initial opinion independently of its weights.
            silence_mode, dynamic_influence = agent_types[int(type_generator.integers(len(agent_types)))]

            # Beta(0.5, 0.5) favors initial opinions near 0 and 1.
            opinion = float(opinion_generator.beta(opinion_alpha, opinion_beta))

            # Row index receives influence only from itself and its graph neighbors.
            connections = [index, *graph.neighbors(index)]
            weights = np.zeros(n_agents, dtype=np.float64)
            # Draw raw beta weights only on those connections; Agent normalizes the row.
            weights[connections] = weight_generator.beta(weight_alpha, weight_beta, size=len(connections))

            # Agent stores its normalized row and the connections that cannot change later.
            agents.append(Agent(
                index=index,
                opinion=opinion,
                influence_of_others=weights,
                dynamic_influence=dynamic_influence,
                silence_mode=silence_mode,
                tolerance_radius=tolerance_radius if silence_mode != "none" else None,
                majority_threshold=majority_threshold if silence_mode != "none" else None,
            ))

        return cls(agents, distribution_function=distribution_function)

    @property
    def opinion_vector(self) -> np.ndarray:
        """Return each agent's current internal (private) opinion, which keeps evolving even while silent."""
        return np.array([agent.opinion for agent in self.agents], dtype=np.float64)

    @property
    def public_opinion_vector(self) -> np.ndarray:
        """Return each agent's public-opinion value, which can lag behind the internal (private) opinion with memory."""
        return np.array([agent.public_opinion for agent in self.agents], dtype=np.float64)

    @property
    def opinion_participation_vector(self) -> np.ndarray:
        """
        Return a vector of 1s and 0s indicating which agents' public opinions contribute to others' opinion updates; a silent memory agent still contributes with its last public opinion.

        - A DeGroot agent always contributes its current opinion.
        - A memoryless agent contributes its current opinion only while speaking.
        - A memory agent always contributes its last public opinion (while it is speaking, the last public opinion is the current opinion).
        """
        return np.array([agent.opinion_participation for agent in self.agents], dtype=np.float64)

    @property
    def speaking_vector(self) -> np.ndarray:
        """
        Return the speaking state of each agent. 
        
        - A DeGroot agent is always speaking.
        - A memoryless agent is speaking only while it is contributing its current opinion (its opinion_participation and speaking states are the same at all times).
        - A memory agent can be speaking or silent, but it always contributes its last public opinion to others' opinion updates (its opinion_participation is always 1).
        """
        return np.array([agent.speaking for agent in self.agents], dtype=np.bool_)

    @property
    def influence_matrix(self) -> np.ndarray:
        """
        Return a copy of the influence matrix, with row i holding the weights received by agent i from each agent j.
        """
        return np.vstack([agent.influence_of_others for agent in self.agents])

    def calculate_next_opinions(
        self,
        opinions: np.ndarray,
        public_opinions: np.ndarray,
        participation: np.ndarray,
        influences: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate X_i(t+1) = X_i(t) + sum over neighbors j of A_ij(t) S_j(t) [P_j(t) - X_i(t)].
        Expanded: X_i(t+1) = X_i(t) + sum of A_ij(t) S_j(t) P_j(t) - X_i(t) times the sum of A_ij(t) S_j(t).
        opinions contains internal (private) opinions X; public_opinions contains public opinions P.
        """
        active_neighbor_influences = influences.copy()
        np.fill_diagonal(active_neighbor_influences, 0.0)
        active_neighbor_influences *= participation[np.newaxis, :]

        public_term = active_neighbor_influences @ public_opinions
        private_term = opinions * active_neighbor_influences.sum(axis=1)

        return opinions + public_term - private_term

    def calculate_next_speaking(
        self,
        public_opinions: np.ndarray,
        participation: np.ndarray,
    ) -> np.ndarray:
        """
        Decide next speaking states from the current agent opinions and supplied public opinions, excluding self-connections.
        """
        next_speaking = np.empty(self.n_agents, dtype=np.bool_)

        for agent in self.agents:
            neighbors = self._connections[agent.index].copy()
            neighbors[agent.index] = False

            next_speaking[agent.index] = agent.decide_speaking(
                public_opinions[neighbors],
                participation[neighbors],
            )

        return next_speaking

    def calculate_next_influences(
            self, 
            opinions: np.ndarray,
            public_opinions: np.ndarray,
            influences: np.ndarray,
            participation: np.ndarray,
    ) -> np.ndarray:
        """
        Redistribute only the weights of participating neighbors in dynamic rows.
        Silent memoryless neighbors retain their weights until they participate again.
        An agent's distance from itself is always zero, even when its public opinion is outdated.
        """
        next_influences = influences.copy()
        participating = participation.astype(np.bool_)

        for agent in self.agents:
            if not agent.dynamic_influence:
                continue

            neighbors = self._connections[agent.index]
            active_neighbors = neighbors & participating
            active_neighbors[agent.index] = neighbors[agent.index]

            if not np.any(active_neighbors):
                continue

            # Compute the difference between public and private opinions for all positions so agent.index still points to this agent's distance.
            differences = np.abs(public_opinions - opinions[agent.index])
            differences[agent.index] = 0.0
            # Select existing connections only after setting the self-distance to zero.
            values = np.array([self.distribution_function(float(difference)) for difference in differences[active_neighbors]], dtype=np.float64)

            if not np.all(np.isfinite(values)) or np.any(values <= 0):
                raise ValueError("distribution_function must return finite, positive values for existing connections")

            total = float(np.sum(values))

            if not np.isfinite(total) or total <= 0:
                raise ValueError("sum of distribution values must be finite and positive")

            # The active weights sum to one minus the unchanged weights of silent memoryless neighbors.
            remaining_weight = float(np.sum(influences[agent.index, active_neighbors]))
            normalized = remaining_weight * values / total

            if np.any(normalized <= 0):
                raise ValueError("distribution values are too small to preserve every connection")

            next_influences[agent.index, active_neighbors] = normalized

        return next_influences

    def advance_round(self) -> None:
        """
        Calculate opinions, speaking states, and next influence weights from one frozen round, then apply the completed results.
        """
        opinions = self.opinion_vector
        public_opinions = self.public_opinion_vector
        participation = self.opinion_participation_vector
        influences = self.influence_matrix

        next_opinions = self.calculate_next_opinions(opinions, public_opinions, participation, influences)
        next_speaking = self.calculate_next_speaking(public_opinions, participation)
        next_influences = self.calculate_next_influences(opinions, public_opinions, influences, participation)

        if not np.all(np.isfinite(next_opinions)) or np.any((next_opinions < 0) | (next_opinions > 1)):
            raise ValueError("calculated opinions must be in [0, 1]")

        for agent in self.agents:
            if agent.dynamic_influence:
                agent.set_influence_of_others(next_influences[agent.index])
            agent.advance(float(next_opinions[agent.index]), bool(next_speaking[agent.index]))
