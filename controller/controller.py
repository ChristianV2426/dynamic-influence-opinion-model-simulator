from typing import List, Optional
import numpy as np

from model.social_network import SocialNetwork


class Controller:
    """
    Run a social network and retain the states needed to inspect its evolution.

    - social_network: Network whose agents and influence weights evolve each round.
    - n_iterations: Number of opinion-update rounds to execute; the initial state is round zero.
    - record_influence_history: Whether to retain a full influence matrix at every round.
    - record_minimum_influence: Whether to retain the smallest positive influence at every round.
    - consensus_threshold: Optional maximum spread across private and public opinions for early stopping.
    """

    def __init__(
        self,
        social_network: SocialNetwork,
        n_iterations: int,
        record_influence_history: bool = False,
        record_minimum_influence: bool = False,
        consensus_threshold: Optional[float] = None,
    ) -> None:
        if not isinstance(social_network, SocialNetwork):
            raise ValueError("social_network must be a SocialNetwork instance")

        if isinstance(n_iterations, bool) or not isinstance(n_iterations, int) or n_iterations < 0:
            raise ValueError("n_iterations must be a non-negative integer")

        if not isinstance(record_influence_history, bool) or not isinstance(record_minimum_influence, bool):
            raise ValueError("history options must be booleans")

        if consensus_threshold is not None:
            if isinstance(consensus_threshold, (bool, np.bool_)):
                raise ValueError("consensus_threshold must be a number in [0, 1]")

            try:
                consensus_threshold = float(consensus_threshold)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("consensus_threshold must be a number in [0, 1]") from exc

            if not np.isfinite(consensus_threshold) or not 0 <= consensus_threshold <= 1:
                raise ValueError("consensus_threshold must be a number in [0, 1]")

        self.social_network = social_network
        self.n_iterations = n_iterations
        self.record_influence_history = record_influence_history
        self.record_minimum_influence = record_minimum_influence
        self.consensus_threshold = consensus_threshold
        self.completed_iterations = 0
        self.stop_reason: Optional[str] = None

        self.opinion_history: List[np.ndarray] = []
        self.public_opinion_history: List[np.ndarray] = []
        self.speaking_history: List[np.ndarray] = []
        self.influence_matrix_history: List[np.ndarray] = []
        self.minimum_influence_history: List[float] = []

        self._record_state()

    def _record_state(self) -> None:
        """Store independent snapshots of the current state after a completed round, including round zero."""
        self.opinion_history.append(self.social_network.opinion_vector.copy())
        self.public_opinion_history.append(self.social_network.public_opinion_vector.copy())
        self.speaking_history.append(self.social_network.speaking_vector.copy())

        if self.record_influence_history or self.record_minimum_influence:
            influences = self.social_network.influence_matrix

            if self.record_influence_history:
                self.influence_matrix_history.append(influences.copy())

            if self.record_minimum_influence:
                self.minimum_influence_history.append(float(np.min(influences[influences > 0])))

    def _has_consensus(self) -> bool:
        """Check whether private and public opinions fit within one interval of the configured width."""
        opinions = self.opinion_history[-1]
        public_opinions = self.public_opinion_history[-1]
        maximum = max(float(np.max(opinions)), float(np.max(public_opinions)))
        minimum = min(float(np.min(opinions)), float(np.min(public_opinions)))
        return maximum - minimum <= self.consensus_threshold

    def run_simulation(self) -> None:
        """Advance until consensus or the iteration limit, recording the state after each round."""
        if self.consensus_threshold is not None and self._has_consensus():
            self.stop_reason = "consensus"
            return

        while self.completed_iterations < self.n_iterations:
            self.social_network.advance_round()
            self.completed_iterations += 1
            self._record_state()

            if self.consensus_threshold is not None and self._has_consensus():
                self.stop_reason = "consensus"
                return

        self.stop_reason = "iteration_limit"
