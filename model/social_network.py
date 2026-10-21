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
