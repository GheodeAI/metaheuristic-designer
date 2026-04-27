# Metaheuristic‑designer

[![Documentation Status](https://readthedocs.org/projects/metaheuristic-designer/badge/?version=latest)](https://metaheuristic-designer.readthedocs.io/en/latest/?badge=latest)

Object‑oriented framework for the development, testing and analysis of metaheuristic optimization algorithms aimed at researchers and engineers working on optimization problems.

It was inspired by the article [Metaheuristics "in the large"](https://doi.org/10.1016/j.ejor.2021.05.042) that discusses some of the issues in the research on metaheuristic optimization, suggesting the development of libraries for the standardization of metaheuristic algorithms.

Most of the design decisions are based on the book [Introduction to Evolutionary Computing](https://doi.org/10.1007/978-3-662-44874-8) by Eiben and Smith, which is very well explained and is highly recommended to anyone willing to learn about the topic.

This framework doesn't claim to have high performance, especially since the chosen language is Python and the code has not been designed for speed. This  is rarely an issue since the highest amount of time spent in these kind of algorithms tends to be in the evaluation of the objective function. If you want to compare an algorithm made with this tool with another one that is available by other means, it is recommended to use the number of evaluations of the objective function as a metric instead of execution time.

---

## Installation

Install the core package:

```bash
pip install metaheuristic-designer
```

---

## Quick Start

Optimise the classical *sphere* function with a (1+1)‑Evolution Strategy:

```python
import numpy as np
from metaheuristic_designer import *
from metaheuristic_designer.benchmarks import Sphere
from metaheuristic_designer.strategies import HillClimb
from metaheuristic_designer.initializers import UniformInitializer
from metaheuristic_designer.operators import create_operator
from metaheuristic_designer.algorithms import StandardAlgorithm

# Define the problem
objfunc = Sphere(dim=5, mode="min")

# Create an initialiser
init = UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim,
                         pop_size=1, random_state=42)

# Build a mutation operator
mutation = create_operator("mutation.gaussian_mutation", F=0.5, N=1, random_state=42)

# Assemble the search strategy (Hill‑Climbing)
strategy = HillClimb(init, mutation)

# Wrap it in an algorithm and run
alg = StandardAlgorithm(objfunc, strategy, stop_cond="neval", neval=1000, verbose=True)
population = alg.optimize()
best_sol, best_fit = population.best_solution()
print(f"Best fitness: {best_fit}")
```

More examples are available in the `examples/` directory.

---

## Design Overview

The framework is built around a small number of abstract, composable pieces.

### Algorithm
An `Algorithm` orchestrates the optimization loop — initialization, stopping conditions, progress tracking, and logging. The concrete `StandardAlgorithm` implements the classic parent‑selection → perturbation → evaluation → survivor‑selection loop. `MemeticAlgorithm` supports hybrid schemes that embed a local search step.

### Search Strategy
A `SearchStrategy` defines how the population evolves each generation. It holds an initializer, an operator, and optionally parent and survivor selection methods. Pre‑built strategies include:

- `HillClimb`, `LocalSearch`, `SA` (Simulated Annealing)
- `GA`, `ES`, `DE`, `PSO`
- `CrossEntropyMethod`, `GaussianUMDA`, `GaussianPBIL`, `BernoulliUMDA`, `BernoulliPBIL`
- `RandomSearch`, `NoSearch`

All strategies can be extended or replaced with custom implementations.

### Operators
Operators modify the genotype of individuals. They are created through **factories** that accept string keys and optional parameters:

```python
create_operator("mutation.gaussian_mutation", F=0.2, N=3, random_state=42)
create_operator("crossover.one_point_crossover", random_state=42)
create_operator("de.best.1", F=0.8, Cr=0.9)
create_operator("permutation.swap", N=2)
```

A generic factory supports dot‑notation (`"mutation.gaussian_mutation"`, `"crossover.one_point_crossover"`) and runtime registration of new operators via `add_operator_entry`.

### Selection Methods
Parent and survivor selection are handled by separate interfaces (`ParentSelection`, `SurvivorSelection`) and created through dedicated factories:

```python
create_parent_selection("tournament", amount=20, tournament_size=3)
create_survivor_selection("(m+n)")
create_survivor_selection("elitism", amount=5)
```

Available keys include `"best"`, `"random"`, `"roulette"`, `"sus"`, `"one_to_one"`, `"prob_one_to_one"`, `"(m+n)"`, `"(m,n)"`, `"elitism"`, `"cond_elitism"`, `"generational"`, and others. The complete list is documented in the documentation page.

### Population
A `Population` holds the genotype matrix, fitness values, a record of the best individual, and per‑spot historical bests. Methods for slicing, joining, and updating the solution matrix are provided.

### Encodings
An `Encoding` translates between the internal genotype and the phenotype evaluated by the objective function. Built‑in encodings include:

- `DefaultEncoding` (identity)
- `TypeCastEncoding` (convert between float / int / bool)
- `MatrixEncoding` / `ImageEncoding`
- `SigmoidEncoding` (binary problems with continuous operators)
- `ParameterExtendingEncoding` — attach extra per‑individual parameters (e.g., velocity for PSO, sigma for self‑adaptation, ...)

### Initializers
Initializers generate the starting population. Examples: `UniformInitializer`, `GaussianInitializer`, `PermInitializer`, `SeedDetermInitializer`, and `ExtendedInitializer` (for parameter‑extending encodings).

### Benchmarks
A collection of test problems is included:

- Continuous — `Sphere`, `Rastrigin`, `Rosenbrock`, `Ackley`, `Griewank`, `Weierstrass`, etc.
- Binary — `MaxOnes`, `BinKnapsack`, `ThreeSAT`
- Permutation — `MaxClique`
- Image approximation — `ImgApprox`, `ImgEntropy`, `ImgStd`

### Constraint Handling
Constraint handlers implement repair or penalty strategies: `ClipBoundConstraint`, `BounceBoundConstraint`, `CycleBoundConstraint`, `CompositeConstraint`, and linear penalty methods. They can be composed and applied automatically during the optimization loop.

### Parameter Scheduling
Any numeric parameter of an operator, selection method, or strategy can be a **schedulable** value (e.g., a decay schedule). The library provides `LinearSchedule`, `ExponentialSchedule`, `LogisticSchedule`, `StepSchedule`, and `RandomSchedule`. Callable values are evaluated each generation via the `ParametrizableMixin`.

Current parameter values can be obtained by calling `.get_params()` on any class inheriting from the mixin; it returns a dictionary mapping parameter names to their current values. You can also access individual parameters directly via the `.params` attribute, e.g., component.params.scale

---

## Extending the Framework

All concrete classes inherit from abstract bases, making it straightforward to add custom components:

- **Custom Operator** — subclass `Operator` or implement `my_operator(population_matrix, fitness, random_state, **kwargs)` and register it with `add_operator_entry`.
- **Custom Selection** — subclass `ParentSelection` or `SurvivorSelection`, or use the `*FromLambda` wrappers.
- **Custom Strategy** — subclass `SearchStrategy`and override the methods you need (`initialize`, `select_parents`, `perturb`, `select_individuals`, `step`) or use the `SearchStrategyFromLambda` wrappers.
- **Custom Encoding / Initializer / Constraint Handler** — implement the corresponding abstract class or use the `*FromLambda` wrappers.

If you subclass any component, remember to call the parent’s implementation inside your implementation. For instance, if you override `encode(solutions)` in a custom encoding, always include `super().encode(solutions)` to keep the built‑in validation and bookkeeping logic intact.

---

## Documentation

Full API documentation is available at [https://metaheuristic-designer.readthedocs.io](https://metaheuristic-designer.readthedocs.io).

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.