import json
import math
from pathlib import Path
from typing import Any, Callable

from controller.controller import Controller
from model.agent import Agent
from model.social_network import SocialNetwork


def _gaussian_distribution(data: dict[str, Any]) -> Callable[[float], float]:
    """Build f(d) = exp(-gamma * d²) from its JSON parameters."""
    SimulationParser._fields(data, {"type", "gamma"}, "distributionFunction")
    gamma = SimulationParser._number(
        SimulationParser._required(data, "gamma", "distributionFunction"),
        "distributionFunction.gamma",
    )
    if gamma <= 0:
        raise ValueError("distributionFunction.gamma must be positive")

    def gaussian(distance: float) -> float:
        return math.exp(-gamma * distance ** 2)

    return gaussian


class SimulationParser:
    """Build a simulation from a JSON configuration."""

    # Add new JSON distribution types here, each with a parameter-validating builder.
    DISTRIBUTION_FUNCTIONS = {"gaussian": _gaussian_distribution}

    COMMON_FIELDS = {"mode", "iterationLimit", "consensusThreshold", "recordInfluenceHistory",
                     "recordMinimumInfluence", "distributionFunction",}

    CUSTOM_FIELDS = COMMON_FIELDS | {"agents", "edges"}

    RANDOM_FIELDS = COMMON_FIELDS | {
        "nAgents", "neighborsPerAgent", "rewiringProbability", "seed",
        "allowedAgentTypes", "opinionAlpha", "opinionBeta", "weightAlpha",
        "weightBeta", "toleranceRadius", "majorityThreshold",
    }

    @classmethod
    def from_file(cls, path: str | Path) -> Controller:
        """Read a JSON file and construct its agents, network, and controller."""
        with open(path, encoding="utf-8") as file:
            configuration = json.load(file)
        return cls.from_dict(configuration)

    @classmethod
    def from_dict(cls, configuration: dict[str, Any]) -> Controller:
        """Validate a decoded configuration and construct the simulation."""
        cls._object(configuration, "configuration")
        mode = cls._required(configuration, "mode", "configuration")

        if mode == "custom":
            cls._fields(configuration, cls.CUSTOM_FIELDS, "configuration")
            network = cls._custom_network(configuration)
        elif mode == "random":
            cls._fields(configuration, cls.RANDOM_FIELDS, "configuration")
            network = cls._random_network(configuration)
        else:
            raise ValueError("mode must be 'custom' or 'random'")

        return Controller(
            social_network=network,
            n_iterations=cls._required(configuration, "iterationLimit", "configuration"),
            consensus_threshold=configuration.get("consensusThreshold"),
            record_influence_history=configuration.get("recordInfluenceHistory", False),
            record_minimum_influence=configuration.get("recordMinimumInfluence", False),
        )

    @classmethod
    def _custom_network(cls, configuration: dict[str, Any]) -> SocialNetwork:
        agent_data = cls._required(configuration, "agents", "configuration")
        edge_data = cls._required(configuration, "edges", "configuration")

        if not isinstance(agent_data, list) or not agent_data:
            raise ValueError("agents must be a non-empty list")
        if not isinstance(edge_data, list):
            raise ValueError("edges must be a list")

        names = []
        for index, data in enumerate(agent_data):
            label = f"agents[{index}]"
            cls._object(data, label)
            cls._fields(data, {
                "name", "opinion", "dynamicInfluence", "silenceMode",
                "toleranceRadius", "majorityThreshold",
            }, label)

            name = cls._required(data, "name", label)
            if not isinstance(name, str) or not name.strip() or name in names:
                raise ValueError(f"{label}.name must be a unique, non-empty string")
            names.append(name)

        indices = {name: index for index, name in enumerate(names)}
        weights = [[0.0] * len(names) for _ in names]
        seen_edges = set()

        for index, data in enumerate(edge_data):
            label = f"edges[{index}]"
            cls._object(data, label)
            cls._fields(data, {"source", "target", "influence"}, label)
            source = cls._required(data, "source", label)
            target = cls._required(data, "target", label)

            if (not isinstance(source, str) or not isinstance(target, str)
                    or source not in indices or target not in indices):
                raise ValueError(f"{label} must refer to agent names in source and target")
            
            if (source, target) in seen_edges:
                raise ValueError(f"{label} duplicates the edge from {source} to {target}")
            
            seen_edges.add((source, target))

            influence = cls._number(cls._required(data, "influence", label), f"{label}.influence")
            if influence < 0:
                raise ValueError(f"{label}.influence must be non-negative")
            
            # source influences target: row target receives weight from column source.
            weights[indices[target]][indices[source]] = influence

        if any(not any(weight > 0 for weight in row) for row in weights):
            raise ValueError("every agent must have at least one positive incoming influence")

        agents = []
        for index, data in enumerate(agent_data):
            opinion = cls._number(cls._required(data, "opinion", f"agents[{index}]"), f"agents[{index}].opinion")

            agents.append(Agent(
                index=index,
                name=names[index],
                opinion=opinion,
                influence_of_others=weights[index],
                dynamic_influence=data.get("dynamicInfluence", False),
                silence_mode=data.get("silenceMode", "none"),
                tolerance_radius=data.get("toleranceRadius"),
                majority_threshold=data.get("majorityThreshold"),
            ))

        dynamic = any(agent.dynamic_influence for agent in agents)

        return SocialNetwork(agents, distribution_function=cls._distribution_function(configuration, dynamic))

    @classmethod
    def _random_network(cls, configuration: dict[str, Any]) -> SocialNetwork:
        types = configuration.get("allowedAgentTypes")

        if types is not None:
            if not isinstance(types, list) or not types:
                raise ValueError("allowedAgentTypes must be a non-empty list")
            
            allowed_types = []

            for index, data in enumerate(types):
                label = f"allowedAgentTypes[{index}]"
                cls._object(data, label)
                cls._fields(data, {"silenceMode", "dynamicInfluence"}, label)
                allowed_types.append((
                    cls._required(data, "silenceMode", label),
                    cls._required(data, "dynamicInfluence", label),
                ))
        else:
            allowed_types = None

        dynamic = allowed_types is not None and any(isinstance(agent_type[1], bool) and agent_type[1] for agent_type in allowed_types)

        return SocialNetwork.generate_random_network(
            n_agents=cls._required(configuration, "nAgents", "configuration"),
            neighbors_per_agent=cls._required(configuration, "neighborsPerAgent", "configuration"),
            rewiring_probability=cls._required(configuration, "rewiringProbability", "configuration"),
            seed=configuration.get("seed"),
            allowed_agent_types=allowed_types,
            opinion_alpha=configuration.get("opinionAlpha", 0.5),
            opinion_beta=configuration.get("opinionBeta", 0.5),
            weight_alpha=configuration.get("weightAlpha", 0.5),
            weight_beta=configuration.get("weightBeta", 0.5),
            tolerance_radius=configuration.get("toleranceRadius"),
            majority_threshold=configuration.get("majorityThreshold"),
            distribution_function=cls._distribution_function(configuration, dynamic),
        )

    @classmethod
    def _distribution_function(cls, configuration: dict[str, Any], dynamic: bool) -> Callable[[float], float] | None:
        data = configuration.get("distributionFunction")

        if data is None:
            if dynamic:
                raise ValueError("distributionFunction is required for dynamic agents")
            return None

        cls._object(data, "distributionFunction")
        function_type = cls._required(data, "type", "distributionFunction")
        
        if not isinstance(function_type, str) or function_type not in cls.DISTRIBUTION_FUNCTIONS:
            raise ValueError(f"distributionFunction.type must be one of {tuple(cls.DISTRIBUTION_FUNCTIONS)}")

        return cls.DISTRIBUTION_FUNCTIONS[function_type](data)

    @staticmethod
    def _object(value: Any, label: str) -> None:
        if not isinstance(value, dict):
            raise ValueError(f"{label} must be an object")

    @staticmethod
    def _fields(value: dict[str, Any], allowed: set[str], label: str) -> None:
        unexpected = value.keys() - allowed
        if unexpected:
            raise ValueError(f"{label} has unknown fields: {', '.join(sorted(unexpected))}")

    @staticmethod
    def _required(value: dict[str, Any], field: str, label: str) -> Any:
        if field not in value:
            raise ValueError(f"{label}.{field} is required")
        return value[field]

    @staticmethod
    def _number(value: Any, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} must be a finite number")
        
        try:
            number = float(value)
        except OverflowError as exc:
            raise ValueError(f"{label} must be a finite number") from exc
        
        if not math.isfinite(number):
            raise ValueError(f"{label} must be a finite number")
        
        return number
