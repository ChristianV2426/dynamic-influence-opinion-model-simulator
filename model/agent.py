from typing import Optional, Sequence
import numpy as np


class Agent:
    """
    Represent one agent's opinion, influences, and silence behavior.

    - index: Zero-based position of this agent in the network and its vectors.
    - opinion: Initial internal opinion in [0, 1]; also the initial public opinion.
    - influence_of_others: One weight per network agent; entry j is how much agent j influences this agent, including self-influence at entry index. Weights must be non-negative and are normalized to sum to one; an all-zero row becomes full self-influence.
    - name: Optional label; defaults to the string form of index.
    - dynamic_influence: Whether the network may update this agent's influence row between rounds; the update rule belongs to the network, not this class.
    - silence_mode: "none" keeps the agent speaking; "memoryless" lets it stop contributing to opinion updates while its public opinion still follows its internal opinion; "memory" keeps its last public opinion available while it is silent.
    - tolerance_radius: Maximum opinion distance for a neighbor to count as close, in [0, 1]; required for either silence mode.
    - majority_threshold: Minimum fraction of considered neighbors that must be close for the agent to speak, in [0, 1]; required for either silence mode.
    """

    SILENCE_MODES = ("none", "memoryless", "memory")

    def __init__(
        self,
        index: int,
        opinion: float,
        influence_of_others: Sequence[float],
        name: Optional[str] = None,
        dynamic_influence: bool = False,
        silence_mode: str = "none",
        tolerance_radius: Optional[float] = None,
        majority_threshold: Optional[float] = None,
    ) -> None:
        if not isinstance(index, int) or index < 0:
            raise ValueError("index must be a non-negative integer")
        
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise ValueError("name must be a non-empty string")
        
        if not isinstance(dynamic_influence, bool):
            raise ValueError("dynamic_influence must be a boolean")
        
        if silence_mode not in self.SILENCE_MODES:
            raise ValueError(f"silence_mode must be one of {self.SILENCE_MODES}")

        self.index = index
        self.name = name if name is not None else str(index)
        self.dynamic_influence = dynamic_influence
        self.silence_mode = silence_mode

        self.tolerance_radius = (None if tolerance_radius is None
                                 else self._unit_interval(tolerance_radius, "tolerance_radius"))
        
        self.majority_threshold = (None if majority_threshold is None
                                   else self._unit_interval(majority_threshold, "majority_threshold"))
        
        if silence_mode != "none" and (self.tolerance_radius is None or self.majority_threshold is None):
            raise ValueError("silence agents need tolerance_radius and majority_threshold")

        self._opinion = self._unit_interval(opinion, "opinion")
        self._public_opinion = self._opinion
        self._speaking = True

        self._influence_of_others = self._normalized_weights(influence_of_others)

        if index >= self._influence_of_others.size:
            raise ValueError("index must refer to an entry in influence_of_others")
        
        self._initial_connections = self._influence_of_others > 0

    @staticmethod
    def _unit_interval(value: float, name: str) -> float:
        """
        Validate that a value is a number in [0, 1] and return it as a float.
        """
        if value is None or isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be a number in [0, 1]")
        
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a number in [0, 1]") from exc

        
        if not np.isfinite(number) or not 0 <= number <= 1:
            raise ValueError(f"{name} must be a number in [0, 1]")
        
        return number

    @staticmethod
    def _positive_shape_parameter(value: float, name: str) -> float:
        """Validate a positive, finite distribution shape parameter."""
        if value is None or isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be a finite, positive number")

        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a finite, positive number") from exc

        if not np.isfinite(number) or number <= 0:
            raise ValueError(f"{name} must be a finite, positive number")

        return number

    def _normalized_weights(self, weights: Sequence[float]) -> np.ndarray:
        """
        Return a normalized float64 row, using self-influence for a zero row.
        """
        try:
            row = np.asarray(weights, dtype=np.float64).copy()
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("influence_of_others must contain numeric weights") from exc

        if row.ndim != 1 or row.size == 0:
            raise ValueError("influence_of_others must be a non-empty 1-D vector")
        
        if not np.all(np.isfinite(row)) or np.any(row < 0):
            raise ValueError("influence weights must be finite and non-negative")
        
        total = float(np.sum(row))

        if not np.isfinite(total):
            raise ValueError("sum of influence weights must be finite")
        
        if total == 0:
            if self.index >= row.size:
                raise ValueError("index must refer to an entry in influence_of_others")
            row[self.index] = 1.0

        else:
            row /= total

        return row

    @property
    def opinion(self) -> float:
        return self._opinion

    @property
    def public_opinion(self) -> float:
        return self._public_opinion

    @property
    def speaking(self) -> bool:
        return self._speaking

    @property
    def opinion_participation(self) -> int:
        """
        Return the factor used for this agent in the opinion update.
        - A DeGroot agent always contributes its current opinion.
        - A memoryless agent contributes its current opinion only while speaking.
        - A memory agent always contributes its last public opinion (while it is speaking, the last public opinion is the current opinion).
        """
        return int(self.silence_mode != "memoryless" or self._speaking)

    @property
    def influence_of_others(self) -> np.ndarray:
        return self._influence_of_others.copy()

    def set_influence_of_others(self, weights: Sequence[float]) -> None:
        """
        Replace this row without changing its original connections.
        """
        row = self._normalized_weights(weights)

        if row.size != self._influence_of_others.size:
            raise ValueError("the influence row cannot change length")
        
        if not np.array_equal(row > 0, self._initial_connections):
            raise ValueError("the influence row cannot create or remove connections")
        
        self._influence_of_others = row

    def decide_speaking(
        self,
        neighbor_public_opinions: Sequence[float],
        neighbor_opinion_participation: Sequence[bool],
    ) -> bool:
        """
        Decide the next speaking state from one unchanged round snapshot.

        The network supplies this agent's neighbors, excluding itself, with each
        neighbor's public opinion and opinion_participation. A silent memoryless
        neighbor is excluded; a silent memory neighbor remains represented by
        its last public opinion. With no considered neighbors, the paper's
        majority inequality is satisfied (0 >= 0).
        """
        if self.silence_mode == "none":
            return True

        opinions = np.asarray(neighbor_public_opinions, dtype=np.float64)
        participation_values = np.asarray(neighbor_opinion_participation)

        if opinions.ndim != 1 or participation_values.ndim != 1 or opinions.size != participation_values.size:
            raise ValueError("neighbor opinions and participation states must be equal-length vectors")
        
        if not np.all(np.isfinite(opinions)) or np.any((opinions < 0) | (opinions > 1)):
            raise ValueError("neighbor opinions must be in [0, 1]")
        
        if not np.all(np.isin(participation_values, (False, True))):
            raise ValueError("neighbor participation states must be booleans")
        
        considered = participation_values.astype(np.bool_)

        close = np.abs(opinions - self._opinion) <= self.tolerance_radius

        return bool(np.count_nonzero(close & considered)
                    >= self.majority_threshold * np.count_nonzero(considered))

    def advance(self, new_opinion: float, next_speaking: bool) -> None:
        """
        Commit this agent's next opinion and speaking state together.

        DeGroot and memoryless agents expose their current opinion every round.
        A memory agent retains its last public opinion while it is silent.
        """
        opinion = self._unit_interval(new_opinion, "new_opinion")

        if not isinstance(next_speaking, (bool, np.bool_)):
            raise ValueError("next_speaking must be a boolean")
        
        if self.silence_mode == "none" and not next_speaking:
            raise ValueError("an agent without silence must keep speaking")

        self._opinion = opinion
        self._speaking = bool(next_speaking)

        if self.silence_mode != "memory" or self._speaking:
            self._public_opinion = opinion

    @classmethod
    def generate_random_agent(
        cls,
        index: int,
        n_agents: int,
        rng: Optional[np.random.Generator] = None,
        opinion_alpha: float = 0.5,
        opinion_beta: float = 0.5,
        weight_alpha: float = 0.5,
        weight_beta: float = 0.5,
        **agent_options,
    ) -> "Agent":
        """
        Generate an opinion and one influence row with a supplied RNG.

        - opinion_alpha, opinion_beta: Beta shape parameters; the default 0.5/0.5 favors opinions near 0 and 1.
        - weight_alpha, weight_beta: Beta shape parameters for independent raw influence draws; the default 0.5/0.5 favors values near 0 and 1 before normalization.

        This method draws one raw weight for every agent, including self-influence.
        Agent normalizes the row afterward, so its final weights do not follow
        the raw beta distribution.
        """
        if isinstance(n_agents, bool) or not isinstance(n_agents, int) or n_agents < 1:
            raise ValueError("n_agents must be a positive integer")

        opinion_alpha = cls._positive_shape_parameter(opinion_alpha, "opinion_alpha")
        opinion_beta = cls._positive_shape_parameter(opinion_beta, "opinion_beta")
        weight_alpha = cls._positive_shape_parameter(weight_alpha, "weight_alpha")
        weight_beta = cls._positive_shape_parameter(weight_beta, "weight_beta")
        
        generator = rng if rng is not None else np.random.default_rng()
        # Beta(0.5, 0.5) draws more often near 0 or 1 than near the center.
        opinion = float(generator.beta(opinion_alpha, opinion_beta))
        # Draw raw weights independently. Agent normalizes their row to sum to one.
        weights = generator.beta(weight_alpha, weight_beta, size=n_agents)

        return cls(index, opinion, weights, **agent_options)
