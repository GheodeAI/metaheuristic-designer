# Metaheuristic‑designer

[![PyPI - Version](https://img.shields.io/pypi/v/metaheuristic-designer)](https://pypi.org/project/metaheuristic-designer/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/metaheuristic-designer)](https://pypi.org/project/metaheuristic-designer/)
[![PyPI - Status](https://img.shields.io/pypi/status/metaheuristic-designer)](https://pypi.org/project/metaheuristic-designer/)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/GheodeAI/metaheuristic-designer/python-package.yml)
[![Documentation Status](https://readthedocs.org/projects/metaheuristic-designer/badge/?version=latest)](https://metaheuristic-designer.readthedocs.io/en/latest/?badge=latest)

**A modular, object‑oriented framework for building, testing, and analysing
population‑based optimisation algorithms.**  
Designed for researchers who demand **reproducible, composable, and
rigorously comparable** metaheuristics – and for practitioners who want
to get results quickly without sacrificing control.

The library is a direct implementation of the vision laid out in
[Metaheuristics “In the Large”](https://doi.org/10.1016/j.ejor.2021.05.042).
Many design decisions follow the principles of
[Introduction to Evolutionary Computing](https://doi.org/10.1007/978-3-662-44874-8)
by Eiben and Smith.

---

## Installation

```bash
pip install metaheuristic-designer
```

To use the plotting examples, install the optional dependencies:

```bash
pip install metaheuristic-designer[examples]
```

---

## Why metaheuristic‑designer?

- **Reproducibility by default** – Every component accepts a `rng`
  (a NumPy Generator or a seed).  Passing the same seed guarantees
  identical runs, making your experiments truly reproducible.
- **Truly composable** – Algorithms are built by plugging together
  abstract interfaces (`Initializer`, `Encoding`, `Operator`,
  `ParentSelection`, `SurvivorSelection`, `SearchStrategy`).  Any piece
  can be swapped independently, enabling systematic exploration of
  algorithm design space.
- **Transparent and open** – The architecture hides no hidden mechanisms.
  Every operator, selection method, and schedule is explicitly configured,
  and the source code is designed to be read and extended.
- **Built‑in parameter scheduling** – Mutation strengths, probabilities,
  temperatures – any numeric parameter can follow a schedule that adapts
  during the search, giving you fine‑grained control over exploration
  vs. exploitation.
- **Rich analysis capabilities** – Track best, median, worst, full
  population fitness, diversity, and scheduled parameter values across
  generations.  The recorded data is easily exported to a pandas DataFrame
  for plotting with your favourite library (seaborn, matplotlib, plotly, …).
- **Safe checkpointing** – The algorithm can periodically save its entire
  state to disk.  If the process is interrupted (Ctrl+C, SIGTERM), the
  latest checkpoint is automatically preserved.  You can resume the run
  later without losing any progress.
- **Modular monitoring** – Choose between a silent run, a `tqdm` progress
  bar, or periodic text reports.  The reporting interface is extensible:
  you can write your own reporter to log custom metrics or integrate with
  external dashboards.
- **Research‑grade yet beginner‑friendly** – Start with the
  `metaheuristic_designer.simple` module for one‑line instantiation of
  common algorithms (GA, DE, PSO, SA, …).  When you need more control,
  dive into the object‑oriented configuration.

---
## Tutorials

The best way to learn is to run the interactive notebooks – they cover everything from basic setup to
advanced self‑adaptation and live plotting. Each notebook is a self‑contained, commented
Python script (Jupytext format) that you can open as a notebook or run as a script.

| Notebook | What you’ll learn |
|----------|--------------------|
| [Genetic Algorithm Quickstart](tutorials/genetic_algorithm_tutorial.ipynb) | Minimise the Sphere function with a GA, convergence plots, log‑scale |
| [Simple API](tutorials/simple_api_tutorial.ipynb) | One‑line GA, DE, PSO; compare algorithms side‑by‑side |
| [Custom Components](tutorials/custom_components_tutorial.ipynb) | Build your own objective, operator, selection, and wiring |
| [Scheduled Parameters](tutorials/parameter_schedules_tutorial.ipynb) | Decay mutation strength, step‑change probabilities, live tracking |
| [Self‑Adapting ES](tutorials/self_adaption_tutorial.ipynb) | Evolution Strategy with evolving sigma, parameter‑extending encodings |
| [Algorithm Selection & Reporting](tutorials/algorithm_selection_tutorial.ipynb) | Run contests, collect raw data, produce statistical reports |
| [Plotting and History Analysis](tutorials/plotting.ipynb) | Fitness distribution boxplots, diversity, scheduled parameter evolution |
| [Real time algorithm progress](tutorials/real_time_plotting_tutorial.ipynb) | Real time demonstration of differential evolution over a 2-D function |
| [Permutation Problems (TSP)](tutorials/tsp_tutorial.ipynb) | PMX crossover, swap mutation, real‑time tour visualisation |

All tutorials are in the `tutorials/` directory of this repository. Open them with Jupyter,
VS Code (Jupyter extension), or any editor that supports Jupytext. If you prefer a live
environment, you can go to [interactive](https://mybinder.org/v2/gh/GheodeAI/metaheuristic-designer/main?filepath=tutorials%2Fgenetic_algorithm_tutorial.py)

---

## Quick Start – minimise the Sphere function with a Genetic Algorithm

```python
import metaheuristic_designer as mhd
from metaheuristic_designer.benchmarks import Sphere
from metaheuristic_designer.strategies import GA
from metaheuristic_designer.initializers import UniformInitializer
from metaheuristic_designer.operators import create_operator
from metaheuristic_designer.parent_selection import create_parent_selection
from metaheuristic_designer.survivor_selection import create_survivor_selection
from metaheuristic_designer.algorithms import Algorithm

# 1. Define the problem (5‑dimensional, minimisation)
objfunc = Sphere(dimension=5, mode="min")

# 2. Create an initializer – random vectors between -10 and 10
rng = mhd.check_rng(42)   # fix the random seed for reproducibility
init = UniformInitializer(objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound,
                          pop_size=100, rng=rng)

# 3. Build the operators (Gaussian mutation + uniform crossover)
mutation = create_operator("mutation.gaussian_mutation", F=0.1, N=1, rng=rng)
crossover = create_operator("crossover.uniform", rng=rng)

# 4. Selection methods
parent_sel = create_parent_selection("tournament", amount=50, tournament_size=3, rng=rng)
survivor_sel = create_survivor_selection("elitism", amount=25, rng=rng)

# 5. Assemble the search strategy (Genetic Algorithm)
strategy = GA(
    initializer=init,
    mutation_op=mutation,
    crossover_op=crossover,
    parent_sel=parent_sel,
    survivor_sel=survivor_sel,
    mutation_prob=0.3,
    crossover_prob=0.9,
    rng=rng,
)

# 6. Run the algorithm for 200 generations
alg = Algorithm(
    objfunc, strategy,
    stop_condition_str="max_iterations",
    max_iterations=200,
    reporter="tqdm",
)
population = alg.optimize()

# 7. Inspect the result
solution, objective = population.best_solution()
print(f"Best objective: {objective:.6g}")
print(f"Decoded solution: {solution}")
```

More interactive examples are available in the `tutorials/` directory, and
executable scripts in the `examples/` directory demonstrate a wide range of
algorithms on classic optimisation tasks.

**Discover all built‑in components** – The complete catalogue of operators,
parent/survivor selection methods, encodings, initializers, and search
strategies – including all accepted parameters – is always up‑to‑date in the
online documentation.  You can also call `list_operators()`,
`list_parent_selection_methods()`, and `list_survivor_selection_methods()`
to print the registered keys directly from Python.

---

## Design Overview

The library is built around a small number of abstract, composable pieces.

By default, every function works with numpy arrays in vectorized operators. `Encodings` allow for non-matrix representations in case it is needed.

### Objective function
Objective function define the optimization problem we want to solve. They will be implemented as `ObjectiveFunc` objects which have an `.objective(solution)` method that evaluates each solution.

This kind of functions also accept full populations (numpy arrays or iterables) as `.objective(solutions)` returning a vector with the objective of each solution so operations can be vectorized, obtaining very significant performance gains.

It is recommended to define objectives as plain python functions, and pass them to `OperatorFromLambda` to cretate the objective function object.

### Algorithm
An `Algorithm` runs the optimisation loop — initialisation, stopping conditions,
progress tracking, and logging.  It can be configured with object‑oriented
components or with simple keyword arguments.

### Search Strategy (Single iteration)
A `SearchStrategy` defines how the population evolves each generation.
It holds an initializer, an operator, and optionally parent/survivor selection.
Pre‑built strategies include:

- `HillClimb`, `LocalSearch`, `SA` (Simulated Annealing)
- `GA`, `ES`, `DE`, `PSO`
- `CrossEntropyMethod`, `GaussianUMDA`, `GaussianPBIL`, `BernoulliUMDA`, `BernoulliPBIL`, `CMA‑ES`
- `RandomSearch`
- `TabuSearch`, `IteratedLocalSearch` (coming in v1.2)

And hybrid strategies:
- `MemeticStrategy`, (more coming in v1.2)

Custom strategies can be assembled directly with `SearchStrategy` or defined
from scratch.

### Operators (Mutation/crossover)
Operators modify the genotype of individuals.  

This includes both mutation operators, crossover and any other operations that perturb the population in any way.

They are created through a
**factory** that accepts a string key and optional parameters:

```python
create_operator("mutation.gaussian_mutation", F=0.2, N=3, rng=42)
create_operator("crossover.one_point_crossover", rng=42)
create_operator("DE/best/1", F=0.8, Cr=0.9)
create_operator("permutation.swap", N=2)
```

A generic factory supports dot‑notation and runtime registration of custom
operators via `add_operator_entry`.  To see all available operator keys,
call `list_operators()`.

### Selection Methods (Parent/Survivor selection)
Two separate selection steps are distinguished:

- **Parent selection** – chooses which solutions will be used to generate
  new candidates.  Created via `create_parent_selection`.
- **Survivor selection** – decides which solutions survive to the next
  generation.  Created via `create_survivor_selection`.

Both factories accept a method name and optional parameters:

```python
create_parent_selection("tournament", amount=20, tournament_size=3)
create_survivor_selection("(m+n)")
create_survivor_selection("elitism", amount=5)
```

Available keys include `"best"`, `"random"`, `"roulette"`, `"sus"`,
`"one_to_one"`, `"prob_one_to_one"`, `"(m+n)"`, `"(m,n)"`, `"elitism"`,
`"cond_elitism"`, `"generational"`, and more.  To list all registered
methods, use `list_parent_selection_methods()` and
`list_survivor_selection_methods()`.

### Population
A `Population` stores a genotype matrix, fitness & objective values, the
best individual, and per‑spot historical bests.  Use `best_solution()` to
obtain the decoded best solution and its raw objective, and
`best_individual()` to get the genotype and its fitness value.

### Encodings 
An `Encoding` translates between the internal genotype and the phenotype
evaluated by the objective function. 

It effectively mantains a different representation for the solution (the final result of the optimization) and the internal representation during the search.

Built‑in encodings:

- `DefaultEncoding` (identity)
- `TypeCastEncoding` (float / int / bool)
- `MatrixEncoding` / `ImageEncoding`
- `SigmoidEncoding` (binary problems with continuous operators)
- `ParameterExtendingEncoding` — attach extra per‑individual parameters
  (velocity for PSO, sigma for self‑adaptation, …)

### Initializers
Initializers generate the starting population: `UniformInitializer`,
`GaussianInitializer`, `PermInitializer`, `SeededInitializer`,
`DirectInitializer`, and `ExtendedInitializer`
(for parameter‑extending encodings).

### Constraint Handling
Constraint handlers implement repair or penalty strategies:
`ClipBoundConstraint`, `BounceBoundConstraint`, `CycleBoundConstraint`,
`CompositeConstraint`, and linear penalty methods.

### Parameter Scheduling
Any numeric parameter can be a **schedulable** value (e.g., a decay schedule).
The library provides `LinearSchedule`, `LogisticSchedule`, `StepSchedule`, `CosineSchedule`,
`RandomSchedule`, and `ThresholdSchedule`.  Callable values are evaluated
each generation; you can also access current values via `.get_params()` or
the `.params` attribute.

## Algorithm configuration

Algorithms can be configured with different stopping conditions, reporters (real time tracking of the progress), history tracking (what information to collect across iterations) and checkpointing strategies. 

Each will correspond to a different class that is passed to the `Algorithm` class at construction.

### Stopping condition

Stopping conditions are indicated with the `ParsedStoppingCondition` class, which decides when to stop the execution of the algorithm. 

It also has a progress value that is used by some algorithms internally to modify the internal parameters. 

Stopping conditions can be indicated as logical expressions, such as `max_iterations or real_time_limit` (stop when **EITHER** the maximum number of iteratios have passed or a time limit is reached) or `convergence and real_time_limit` (stop when **BOTH** a number of iterations without improvement and a certain time has passed).

### History tracker

History trackers (`ConfigurableHistoryTracker`) store informtion about each iteration for plots and post-execution analysis. 

It can store the best/median/worst solutions and their objective values, diversity metrics and even a historic of the full population and their objective.

### Reporters
Reporters control what information is displayed during a run.  Three
implementations are provided out‑of‑the‑box:

- `TQDMReporter` – a progress bar (requires `tqdm`)
- `VerboseReporter` – periodic text summaries
- `SilentReporter` – no output

You can also create your own reporter by subclassing `Reporter` and
implementing `log_init`, `log_step`, and `log_end`.

### Checkpointer
Long‑running experiments can be protected with the built‑in `PickleCheckpointer`:
- **Periodic saving** – The full algorithm state is automatically saved
  every *N* iterations or after a configurable time interval.
- **OS‑signal safety** – If the process receives a `SIGINT` (Ctrl+C) or
  `SIGTERM` (e.g., from a batch scheduler), the current state is
  immediately dumped to disk before the program exits.  You can resume
  the interrupted run later with
  `checkpointer.load("checkpoint.pkl")` and continue as if nothing happened.
- **Tunable frequency** – Choose between iteration‑based or time‑based
  checkpoints, or use both simultaneously.

This makes metaheuristic‑designer suitable for expensive, long‑running
optimisation tasks on shared clusters or cloud instances where interruptions
are expected.

### Benchmarks
A collection of test problems is included:

- Continuous — `Sphere`, `Rastrigin`, `Rosenbrock`, `Ackley`, `Griewank`,
  `Weierstrass`, etc.
- BBOB benchmarks — `BBOBObjective` 
- IOH wrapper — `IOHObjective`
- Binary — `MaxOnes`, `BinKnapsack`, `ThreeSAT`
- Permutation — `MaxClique`, `TSP`
- Image approximation — `ImgApprox`, `ImgEntropy`, `ImgStd`

---

## Reproducibility and Scientific Rigour

- **Seeded randomness everywhere** – Every random component (initializers,
  operators, selection, even parameter schedules with randomness) is
  driven by a `rng` that you can fix.  Use
  `mhd.check_rng(42)` to get a managed `numpy.random.Generator`.

- **Deterministic experiments** – When you pass the same seed, the entire
  optimisation run, including the initial population, mutation steps, and
  selection events, is bit‑for‑bit identical.

- **Tracking for analysis** – The `ConfigurableHistoryTracker` records per‑generation
  statistics (best, median, worst, diversity, scheduled parameters, full
  fitness vector) into a pandas‑compatible DataFrame.  This lets you produce
  publication‑quality convergence plots, statistical comparisons, and
  parameter‑evolution analyses with tools like seaborn or matplotlib.

- **Algorithm comparison made easy** – The `AlgorithmSelection` and
  `StrategySelection` classes allow you to run multiple algorithms across
  multiple trials and automatically collect their histories, simplifying
  rigorous benchmarking.

- **Open‑source, LGPLv3 licensed** – Use it freely in your research, and
  contribute back if you extend it.

---

## Extending the Framework

All components inherit from abstract bases, making custom additions
straightforward:

- **Custom Operator** — use `OperatorFnDef` (matrix‑level) or
  `OperatorFromLambda` (population‑level), then register with
  `add_operator_entry`.
- **Custom Selection** — subclass `ParentSelection` or `SurvivorSelection`,
  or use `*FromLambda` wrappers.
- **Custom Strategy** — subclass `SearchStrategy` and override the methods
  you need, or use `SearchStrategyFromLambda`.
- **Custom Encoding / Initializer / Constraint Handler** — implement the
  corresponding abstract class or use the `*FromLambda` wrappers.

The documentation includes a full guide on
**Extending the Framework with Custom Components**.

---

## Documentation

The **primary source** for exploring the library’s capabilities is the
online documentation at
[https://metaheuristic-designer.readthedocs.io](https://metaheuristic-designer.readthedocs.io).
It contains:

- A searchable, browsable API reference covering every class and method.
- **Full catalogues** of built‑in operators, selection methods, encodings,
  initializers, search strategies, and benchmarks, with parameter tables.
- Step‑by‑step tutorials (including custom components, parameter schedules,
  self‑adaptive ES, real‑time plotting, and reproducible benchmarking).
- A plotting guide with example visualizations for convergence, diversity,
  fitness distributions, and scheduled parameters.

If you find yourself unsure which operator or selection method to use,
start there – the documentation is designed as a discovery tool as much
as a reference.

---
## License

This framework is distributed under the GNU Lesser General Public License version 3 (LGPL‑v3).  
The full license text is available in the [LICENSE](LICENSE) file.

### What this means for you

- **You are free to use this framework in any project** – academic, commercial, or personal – without opening your own code. You can build your optimisation algorithms, experiments, and proprietary applications on top of it, just like you would with a permissively licensed library.
- **If you modify the framework itself** (its core classes, operators, or infrastructure) and distribute your modified version to others, you must make the source of those changes available under the same LGPL‑v3 license. This keeps the core open and ensures improvements benefit the whole community.
- There is **no warranty** – the software is provided “as is”, as is standard for open‑source research code.

### Why LGPL‑v3?

The LGPL was designed exactly for libraries and frameworks: it protects the shared foundation while giving you complete freedom over the unique work you build on top of it. This is the same licensing model used by **DEAP** (Distributed Evolutionary Algorithms in Python) and many other scientific frameworks. It promotes a healthy ecosystem where the core stays open and everyone – academia and industry alike – can build on it without friction.
