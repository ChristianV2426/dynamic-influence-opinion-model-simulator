# Dynamic Influence Opinion Model Simulator

This repository implements the Dynamic Influence Model developed as part of my undergraduate thesis on opinion evolution in social networks [3]. The model extends the Spiral-of-Silence model proposed by Aranda et al. [2], which in turn extends the DeGroot model [1]. Its contribution is a homophily-based rule that redistributes the influence assigned to neighboring agents over time. The simulator can combine dynamic influence with agent-specific silence behavior in a single configuration, while also reproducing the fixed-influence DeGroot and silence-only models.

Experiments are defined in JSON files, executed from the command line and visualized through the available plotting options.

## Installation

The project requires Python 3.11 or later. Use the Python environment manager of your choice and install the dependencies from the repository root:

```shell
python -m pip install -r requirements.txt
```

## Running a simulation

The command-line entry point is `main.py`. It receives one JSON configuration file:

```shell
python main.py simulations/simulation1.json
```

The default visualization is the evolution of opinions. Several visualizations can be requested together with `--plot`:

```shell
python main.py simulations/simulation7_dynamic.json --plot opinions opinions-with-silence network
```

Available visualizations are:

| Option | Description |
| --- | --- |
| `opinions` | Opinion evolution over time. |
| `opinions-with-silence` | Opinion evolution with the speaking state shown as markers. Non-silent states use circles and silent states use squares. Add `--tolerance-bands` to show tolerance-radius bands around opinions. |
| `private-public` | Opinion and public-opinion evolution. Solid lines represent opinions and dashed lines represent public opinions. The speaking state is marked on the opinion curve. Add `--tolerance-bands` to show tolerance-radius bands around opinions. |
| `network` | Influence graph. When influence history is recorded, the left and right arrow keys navigate through the available times. |
| `minimum-influence` | Evolution of the smallest positive influence, shown on a base-10 logarithmic scale. It requires `recordMinimumInfluence: true`. |

Agent names and colors are shown in the `Agents` legend when the network has at most 20 agents. The legend can be hidden with:

```shell
python main.py simulations/simulation7_dynamic.json --plot private-public --no-agent-legend
```

The `Silent status` and `Plot` legends remain available when they are relevant to the selected visualization.

Tolerance bands show the interval of opinions that each agent considers close to its current opinion, as determined by its tolerance radius. They are disabled by default and are available only for opinion-evolution plots that show silence states: `opinions-with-silence` and `private-public`. Enable them with `--tolerance-bands`:

```shell
python main.py simulations/simulation8_dynamic.json --plot private-public --tolerance-bands
```

## Theoretical model

### DeGroot model

Consider a group of $n$ agents discussing a common proposition. Agent $i$ has an opinion $B_i^t \in [0,1]$ at time $t$, where $0$ represents complete disagreement and $1$ represents complete agreement with the proposition. The opinion state of the network is the vector $B^t = (B_1^t, \ldots, B_n^t) \in [0,1]^n$.

The social network structure of the model is represented by an $n$-agent influence graph, which is a weighted directed graph $G=(A,E,I)$ where:

- $A=\{1,\ldots,n\}$ is the set of agents.
- $E \subseteq A \times A$ is the set of directed influence edges. An edge $(j,i) \in E$ means that source agent $j$ influences receiving agent $i$.
- $I:A \times A \rightarrow [0,1]$ is the influence-weight function. The value $I_{ji}$ is the strength of the influence of agent $j$ on agent $i$, and $I_{ji}=0$ if and only if $(j,i) \notin E$.

For every agent $i$, let $N_i$ denote its neighborhood:

$$
N_i=\{j \in A \setminus \{i\} \mid (j,i) \in E\}.
$$

The influence weights assigned to the neighbors of $i$, together with its self-influence, are normalized:

$$
\sum_{j \in N_i \cup \{i\}} I_{ji}=1,
\qquad \forall i \in A,
$$

where $I_{ii}$ represents the self-influence of agent $i$. The DeGroot opinion update rule is:

$$
B_i^{t+1} = \sum_{j \in N_i \cup \{i\}} I_{ji} B_j^t.
$$

Using the normalization condition, this can equivalently be written in difference form:

$$
B_i^{t+1} = B_i^t + \sum_{j \in N_i} I_{ji}\left(B_j^t - B_i^t\right).
$$

The classical DeGroot model keeps the influence graph and its weights fixed throughout the simulation.

For a fixed influence graph, strong connectivity and aperiodicity are sufficient to guarantee consensus for every initial opinion state under the DeGroot update rule [1]. Strong connectivity ensures that influence can reach every agent through directed paths, while aperiodicity prevents persistent cyclic alternation. Consensus occurs when the agents' opinions converge to a common value.

### Spiral-of-Silence model

Aranda et al. extend the DeGroot framework by allowing agents to decide whether to express their opinions publicly [2]. A Silence Opinion model is defined over an influence graph $G=(A,E,I)$ with an initial opinion state $B^0$, an initial silence state $S^0$, and a state-transition function. At time $t$, its state is the pair $(B^t,S^t)$, where $S^t$ is the silence vector:

$$
S_i^t = 1 \text{ if agent } i \text{ is expressing its opinion},
\qquad
S_i^t = 0 \text{ if agent } i \text{ is silent}.
$$

The model has two versions: a memoryless version, denoted SOM$^-$, and a memory-based version, denoted SOM$^+$. Both use an agent-specific tolerance radius $\tau_i \in [0,1]$ and majority threshold $\mathcal{M}_i \in [0,1]$. Two opinions are close for agent $i$ when:

$$
x \sim_{\tau_i} y
\quad \Longleftrightarrow \quad
|x-y| \leq \tau_i.
$$

#### Memoryless model (SOM$^-$)

In the memoryless version, only the neighbors that are currently speaking contribute to an agent's opinion update. For every agent $i$, the set of speaking neighbors at time $t$ is:

$$
N_i^t = \{j \in N_i \mid S_j^t = 1\}.
$$

The opinion update is:

$$
B_i^{t+1} = B_i^t + \sum_{j \in N_i} I_{ji} S_j^t\left(B_j^t - B_i^t\right).
$$

The silence state for the next time step is determined from the speaking neighborhood:

$$
S_i^{t+1} =
\begin{cases}
1, & \text{if } \mathcal{M}_i |N_i^t|
\leq \left|\{j \in N_i^t \mid B_j^t \sim_{\tau_i} B_i^t\}\right|, \\
0, & \text{otherwise.}
\end{cases}
$$

Thus, agent $i$ speaks at time $t+1$ when at least the fraction $\mathcal{M}_i$ of its speaking neighbors at time $t$ have opinions close to its own.

#### Memory-based model (SOM$^+$)

In the memory-based version, agents initially speak, so $S^0=\mathbf{1}_n$. A silent agent remains publicly represented by the last opinion it expressed. Let $\bar{t}_j$ be the most recent time at or before $t$ when agent $j$ spoke:

$$
\bar{t}_j = \max\{u \leq t \mid S_j^u = 1\}.
$$

The public opinion of agent $j$ at time $t$ is:

$$
\mathrm{pub}B_j^t = B_j^{\bar{t}_j}.
$$

Unlike SOM$^-$, this public opinion remains available while the agent is silent. The opinion update is therefore:

$$
B_i^{t+1} = B_i^t + \sum_{j \in N_i} I_{ji}\left(\mathrm{pub}B_j^t - B_i^t\right).
$$

The silence update evaluates the public opinions of every neighbor in $N_i$:

$$
S_i^{t+1} =
\begin{cases}
1, & \text{if } \mathcal{M}_i |N_i|
\leq \left|\{j \in N_i \mid B_i^t \sim_{\tau_i} \mathrm{pub}B_j^t\}\right|, \\
0, & \text{otherwise.}
\end{cases}
$$

The distinction between the two versions is therefore the availability of a silent neighbor's opinion. SOM$^-$ excludes a silent neighbor from the current update, while in SOM$^+$ the last opinion expressed by that neighbor remains publicly available.

### Dynamic Influence Model

The Dynamic Influence Model proposed in this research [3], extends both the DeGroot model and the Spiral-of-Silence model by allowing influence weights to change over time. This change is governed by a homophily-based redistribution rule: agents assign more influence to neighbors whose opinions are closer to their own. The formulation below first presents the DeGroot-based form of the model.

At time $t$, the network is represented by the influence graph:

$$
G^t=(A,E,I^t).
$$

The agent set $A$ and edge set $E$ remain fixed, whereas the influence function $I^t$ can change over time. The state of the model is the pair $(B^t,I^t)$. The redistribution function $f:[0,1] \rightarrow (0,\infty)$ translates an opinion difference into an unnormalized homophily score. It assigns the greatest score to zero opinion difference and decreases as the opinion difference grows:

$$
f(0)=\max_{\delta \in [0,1]} f(\delta),
\qquad
f'(\delta)<0 \quad \forall \delta \in (0,1).
$$

The opinion update retains the DeGroot difference form, using the influence weights available at that time:

$$
B_i^{t+1}=B_i^t+\sum_{j\in N_i} I_{ji}^t\left(B_j^t-B_i^t\right).
$$

Let $H$ be the set of agents with self-influence:

$$
H=\{i\in A \mid (i,i)\in E\}.
$$

For every receiver $i$ and every source $j\in N_i\cup\{i\mid i\in H\}$, define the absolute opinion difference:

$$
\Delta_{ji}^t=\left|B_j^t-B_i^t\right|.
$$

The influence of $j$ on $i$ for the next time step is then obtained by normalizing the homophily scores within the existing neighborhood of $i$, including self-influence when present:

$$
I_{ji}^{t+1}=
\frac{f\!\left(\Delta_{ji}^t\right)}
{\displaystyle\sum_{k\in N_i\cup\{i\mid i\in H\}}
f\!\left(\Delta_{ki}^t\right)}.
$$

For a fixed receiver $i$, the numerator assigns source $j$ a score based on its opinion distance from $i$. The denominator normalizes those scores over the existing sources of influence on $i$. It cannot be zero because the normalization condition guarantees that this set contains at least one source, and $f$ assigns a strictly positive value to every opinion difference in $[0,1]$.

This rule preserves the topology because it only redistributes influence among existing incoming edges. The resulting model is a homophily-based dynamic influence extension of DeGroot.

Together with an initial opinion state $B^0$ and an initial influence function $I^0$, the opinion and influence update rules determine a simulation run.

### Dynamic Influence Silence Opinion Models

Dynamic influence can be combined with both versions of the Spiral-of-Silence model too. In either case, the state at time $t$ is $(B^t,S^t,I^t)$, where the agent set $A$ and edge set $E$ remain fixed, while opinions, speaking states, and influence weights may change over time.

#### Memoryless dynamic influence model (HDISOM$^-$)

In the memoryless version, a silent neighbor does not contribute to the opinion update. The opinion of agent $i$ evolves according to:

$$
B_i^{t+1}=B_i^t+\sum_{j\in N_i}I_{ji}^tS_j^t\left(B_j^t-B_i^t\right).
$$

The speaking state is determined by the same social-support criterion used in SOM$^-$. The influence update uses the speaking neighborhood $N_i^t$, the opinion difference $\Delta_{ji}^t$, and the self-influence set $H$, all defined above.

For every source $j\in N_i\cup\lbrace i\mid i\in H\rbrace$, the next influence weight is:

$$
I_{ji}^{t+1}=
\begin{cases}
\left(\displaystyle\sum_{k\in N_i^t\cup\lbrace i\mid i\in H\rbrace}I_{ki}^t\right)
\dfrac{f\left(\Delta_{ji}^t\right)}
{\displaystyle\sum_{k\in N_i^t\cup\lbrace i\mid i\in H\rbrace}f\left(\Delta_{ki}^t\right)},
& \text{if } j\in N_i^t\cup\lbrace i\mid i\in H\rbrace,\\
I_{ji}^t,
& \text{if } j\in N_i\setminus N_i^t.
\end{cases}
$$

At each round, the current weights of the speaking neighbors, together with self-influence when it exists, determine the share of agent $i$'s influence subject to redistribution. The rule allocates that share across the eligible connections in proportion to their homophily scores $f\left(\Delta_{ji}^t\right)$, so closer opinions receive larger weights. A silent memoryless neighbor is excluded from this allocation, and its previous weight remains unchanged until it speaks again.

#### Memory-based dynamic influence model (HDISOM$^+$)

The memory-based version uses the public opinions criterion defined for SOM$^+$. Its opinion update is:

$$
B_i^{t+1}=B_i^t+\sum_{j\in N_i}I_{ji}^t\left(\mathrm{pub}B_j^t-B_i^t\right).
$$

The speaking state follows the same social-support rule used in SOM$^+$. The influence redistribution compares the receiving agent's internal opinion with the public opinions available in the network. For every $j\in N_i\cup\{i\mid i\in H\}$, define:

$$
\Delta_{ji}^{t,\mathrm{pub}}=
\begin{cases}
0, & j=i,\\
\left|\mathrm{pub}B_j^t-B_i^t\right|, & j\in N_i.
\end{cases}
$$

The new weights are:

$$
I_{ji}^{t+1}=
\frac{f\left(\Delta_{ji}^{t,\mathrm{pub}}\right)}
{\displaystyle\sum_{k\in N_i\cup\{i\mid i\in H\}}
f\left(\Delta_{ki}^{t,\mathrm{pub}}\right)}.
$$

All existing sources of influence are redistributed in this version, including agents that are currently silent. Their last publicly expressed opinions remain available when the homophily scores are calculated.


## JSON configuration

Every simulation is described by one JSON object. The configuration selects the network source, simulation length, agent behavior, optional histories and optional stopping criteria. The sections below define the common fields and then show how to describe custom and random networks.

Every configuration contains the following top-level fields:

| Field | Required | Description |
| --- | --- | --- |
| `mode` | Yes | `custom` or `random`. |
| `iterationLimit` | Yes | Maximum number of time steps. |
| `consensusThreshold` | No | Optional early stopping threshold based on the spread of opinion and public-opinion values. |
| `recordInfluenceHistory` | No | Records influence matrices for network-history visualization. Defaults to `false`. |
| `recordMinimumInfluence` | No | Records the smallest positive influence at each time step. Defaults to `false`. |
| `distributionFunction` | Conditional | Required when at least one agent has `dynamicInfluence: true`. |

Unknown fields are rejected so that misspelled configuration values are reported immediately.

### Dynamic influence configuration

Set `dynamicInfluence` to `true` for each agent whose influence row should be redistributed during the simulation. The network-level `distributionFunction` field is required whenever at least one agent is dynamic.

The available distribution functions are:

| `type` | Function |
| --- | --- |
| `exponential` | `exp(-c * d)` |
| `gaussian` | `exp(-c * d^2)` |
| `reciprocal-linear` | `1 / (1 + c * d)` |
| `reciprocal-quadratic` | `1 / (1 + c * d^2)` |

Here, `d` is an opinion distance in `[0, 1]` and `c` is a positive concentration parameter. A larger `c` gives a stronger preference to smaller opinion distances.

```json
"distributionFunction": {
  "type": "gaussian",
  "c": 10
}
```

### Custom networks

Use `"mode": "custom"` to define every agent and edge explicitly. The required top-level fields are `agents` and `edges`.

Each agent may define:

| Field | Required | Default | Description |
| --- | --- | --- | --- |
| `name` | Yes | — | Unique identifier used by edges and visualizations. |
| `opinion` | Yes | — | Initial opinion in `[0, 1]`. |
| `dynamicInfluence` | No | `false` | Whether this agent redistributes its influence row over time. |
| `silenceMode` | No | `none` | `none`, `memoryless`, or `memory`. |
| `toleranceRadius` | Conditional | — | Required for a silence mode other than `none`. |
| `majorityThreshold` | Conditional | — | Required for a silence mode other than `none`. |

The `silenceMode` field accepts the following values:

| Value | Meaning |
| --- | --- |
| `none` | The agent is always non-silent. |
| `memoryless` | A silent agent's current opinion is not publicly available for the next opinion update. |
| `memory` | The last opinion expressed by a silent agent remains publicly available for subsequent opinion updates. |

Each edge contains `source`, `target`, and a non-negative `influence`:

```json
{"source": "A", "target": "B", "influence": 0.4}
```

The above edge example means that A influences B. The value is stored in the receiving agent's influence row. Every agent must have at least one positive incoming influence. Rows are normalized by the simulator, and an explicit self-edge should be included when self-influence is desired.

Example of a fixed-influence DeGroot network:

```json
{
  "mode": "custom",
  "iterationLimit": 20,
  "agents": [
    {"name": "A", "opinion": 0.2},
    {"name": "B", "opinion": 0.8}
  ],
  "edges": [
    {"source": "A", "target": "A", "influence": 0.5},
    {"source": "B", "target": "A", "influence": 0.5},
    {"source": "A", "target": "B", "influence": 0.5},
    {"source": "B", "target": "B", "influence": 0.5}
  ]
}
```

### Random networks

Use `"mode": "random"` to generate a connected Watts–Strogatz-like topology together with initial opinions, influence weights and agent behaviors. The generator guarantees connectivity but does not calculate or guarantee formal small-world metrics.

Required fields are:

| Field | Description |
| --- | --- |
| `nAgents` | Number of agents. Must be at least 3. |
| `neighborsPerAgent` | Even number of ring neighbors before rewiring. Must be at least 2 and smaller than `nAgents`. |
| `rewiringProbability` | Rewiring probability in `[0, 1]`. |

Optional fields include a random `seed`, beta-distribution parameters for initial opinions and raw weights, and `allowedAgentTypes`. If more than one agent type is allowed, the generator chooses among them with equal probability. DeGroot without silence is the default type.

### Simulation workflow

The following workflow shows how the configuration fields introduced above fit together when creating and running a new experiment:

1. Set `mode` to `custom` or `random`.
2. Set `iterationLimit` and, if needed, `consensusThreshold`.
3. Define the network and initial opinions through the custom or random-network fields.
4. Select agent behavior. Without additional behavior fields, agents use fixed-influence DeGroot dynamics. Add `dynamicInfluence: true` to the agents that should redistribute their weights. Add `silenceMode`, `toleranceRadius`, and `majorityThreshold` to agents that should use silence behavior.
5. Add `distributionFunction` at the top level if at least one agent is dynamic.
6. Enable `recordInfluenceHistory` or `recordMinimumInfluence` if the corresponding histories are required.
7. Save the JSON file and run it from the repository root with `main.py`.

For example:

```shell
python main.py simulations/simulation7_dynamic.json --plot opinions-with-silence network
```

## Project architecture

The project uses a Model-View-Controller architecture:

- `model/` contains the simulation state and mathematical behavior. `agent.py` represents individual opinions, influence rows, public opinions and silence state. `social_network.py` coordinates matrix-based network updates and dynamic influence redistribution.
- `controller/` contains the simulation orchestration. `simulation_parser.py` validates JSON configurations and constructs the model. `controller.py` advances the simulation and records the histories needed for analysis and visualization.
- `view/` contains the Matplotlib and NetworkX visualizations for opinion evolution, public opinions, silence states, minimum influence and influence networks.
- `simulations/` contains reproducible JSON experiments.
- `main.py` is the command-line entry point that connects the parser, controller, and selected visualizations.

## Included configurations

The `simulations/` directory contains fixed-influence, dynamic-influence, random-network and silence examples. Configurations ending in `_dynamic.json` use dynamic influence, while the other names identify fixed-influence or silence-focused experiments.

Examples:

```shell
python main.py simulations/simulation1.json --plot opinions network
python main.py simulations/simulation2_dynamic.json --plot opinions network minimum-influence
python main.py simulations/simulation7_dynamic.json --plot opinions-with-silence network
```

## References

1. DeGroot, M. H. (1974). [Reaching a Consensus](https://doi.org/10.2307/2285509). *Journal of the American Statistical Association*, 69(345), 118–121.
2. Aranda, J., Díaz, J. F., Gaona, D., and Valencia, F. (2026). [The Spiral of Silence in Multi-agent Models for Opinion Formation](https://doi.org/10.1007/978-3-032-11176-0_24). In *Theoretical Aspects of Computing – ICTAC 2025* (pp. 417–434). Springer Nature Switzerland.
3. Vargas Gutiérrez, C. D. Under the supervision of Aranda, J. and Díaz, J. F. *Opinion Evolution in Social Networks: Extending the Spiral-of-Silence–DeGroot Model with Dynamic Influence Weights*. Undergraduate thesis, Universidad del Valle.
