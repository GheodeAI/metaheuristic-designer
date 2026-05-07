# ARCHITECTURE.md

## 1. Package Objective

`metaheuristic-designer` is a Python framework for building, composing, and evaluating **improvement-based metaheuristic algorithms**. Its central paradigm is the composability of the algorithm lifecycle: every stage—initialization, evaluation, parent selection, perturbation, constraint handling, survivor selection, and progress reporting—is abstracted behind an interface and can be swapped, combined, or replaced at runtime.

The package supports the full spectrum of population-based algorithms: Genetic Algorithms (GA), Differential Evolution (DE), Evolution Strategies (ES, CMA-ES), Particle Swarm Optimization (PSO), Simulated Annealing (SA), Estimation of Distribution Algorithms (EDA), and Bayesian Optimization (BO). It also provides hill-climbing and local-search variants.

---

## 2. General Repository Structure

```
metaheuristic-designer/
├── src/metaheuristic_designer/   # Main package
│   ├── algorithm.py              # Top-level Algorithm orchestrator
│   ├── population.py             # Population data structure
│   ├── objective_function.py     # Objective/fitness abstraction
│   ├── initializer.py            # Initializer abstraction
│   ├── operator.py               # Operator abstraction
│   ├── search_strategy.py        # SearchStrategy abstraction
│   ├── encoding.py               # Encoding abstraction
│   ├── constraint_handler.py     # Constraint handling abstraction
│   ├── parent_selection_base.py  # Parent selection abstraction
│   ├── survivor_selection_base.py# Survivor selection abstraction
│   ├── stopping_condition.py     # Stopping criterion logic
│   ├── history_tracker.py        # Optimization history tracking
│   ├── checkpointer.py           # State persistence
│   ├── reporter.py               # Progress reporting abstraction
│   ├── parametrizable_mixin.py   # Parameter scheduling mixin
│   ├── schedulable_parameter.py  # Schedulable parameter abstraction
│   ├── utils.py                  # Type aliases, JSON encoder, decorators
│   ├── algorithms/               # High-level algorithm wrappers
│   ├── benchmarks/               # Benchmark functions and problems
│   ├── constraint_handlers/      # Concrete constraint strategies
│   ├── encodings/                # Concrete encodings
│   ├── initializers/             # Concrete initializers
│   ├── operators/                # Concrete operators and factories
│   ├── parameter_schedules/      # Concrete schedule implementations
│   ├── parent_selection/         # Concrete parent selection
│   ├── reporters/                # Concrete reporters
│   ├── simple/                   # Convenience constructor functions
│   ├── strategies/               # Concrete search strategies
│   └── survivor_selection/       # Concrete survivor selection
├── tests/                        # Test suite (pytest + hypothesis)
├── docs/                         # Sphinx documentation
├── examples/                     # Usage examples
├── data/                         # TSP/SAT instance files and images
└── pyproject.toml                # Project metadata and tooling config
```

---

## 3. Domain Model

### `Population`
The central data container. Holds the genotype matrix (`n_individuals × dimension`), fitness values, objective values, best-individual tracking, and optionally an `Encoding` for phenotype decoding. All selection and slicing operations return new `Population` instances.

### `ObjectiveFunc`
Abstract base class for the fitness function. Subclasses implement `objective(solution)`. The base class handles constraint penalties, vectorization, and caching (via `recalculate` flag). `VectorObjectiveFunc` adds bound attributes (`lower_bound`, `upper_bound`) to support bound-aware components such as CMA-ES.

### `Algorithm`
Top-level orchestrator. Manages the main optimization loop, stopping conditions, progress reporting, history tracking, and checkpointing. Delegates all search-related logic to the `SearchStrategy`.

### `SearchStrategy`
Defines the algorithmic flow: initialize → evaluate → select parents → perturb → evaluate offspring → repair → select survivors. The concrete hierarchy includes `StaticPopulation`, `VariablePopulation`, `HillClimb`, `LocalSearch`, and strategy-specific subclasses (`GA`, `DE`, `ES`, `CMA_ES`, `SA`, `PSO`, `UMDA`, `PBIL`, `CEM`, `BayesianOptimization`).

### `Operator`
Transforms a population into offspring. Abstract method: `evolve(population, initializer) → Population`. Concrete variants: `NullOperator`, `OperatorFromLambda`, `CompositeOperator`, `BranchOperator`, `MaskedOperator`, `ExtendedOperator`, `AdaptiveOperator`, `BOOperator`.

### `Initializer`
Generates an initial population. Abstract method: `generate_random() → Any`. Variants: `UniformInitializer`, `GaussianInitializer`, `ExponentialInitializer`, `PermInitializer`, `SeedInitializer`, `DirectInitializer`, `ExtendedInitializer`.

### `Encoding`
Maps between genotype (internal representation) and phenotype (solution fed to the objective). Abstract methods: `encode(solutions)` / `decode(population)`. Variants: `DefaultEncoding`, `SigmoidEncoding`, `TypeCastEncoding`, `ImageEncoding`, `MatrixEncoding`, `CompositeEncoding`, `PSO_encoding`, `self_adapting_ES_encoding`, `ParameterExtendingEncoding`.

### `ConstraintHandler`
Manages constraint satisfaction. Abstract methods: `repair_solution(matrix)` / `penalty(matrix)`. Repair-based and penalty-based abstractions exist, along with concrete implementations for bound constraints.

### `ParentSelection` / `SurvivorSelection`
Abstract interfaces for selecting which individuals contribute to the next iteration. Implementations include tournament, roulette wheel, SUS, elitism, and generational replacement.

### `SchedulableParameter` / `ParametrizableMixin`
Any numeric parameter can be replaced with a `SchedulableParameter` (e.g., `LinearSchedule`, `ExponentialDecaySchedule`) that evolves with optimization progress. `ParametrizableMixin` is inherited by all major abstractions; it evaluates callable parameters at each step.

### `StoppingCondition`
Dataclass with multiple criteria: `max_iterations`, `max_evaluations`, `real_time_limit`, `cpu_time_limit`, `objective_target`, `max_patience`. Supports compound boolean expressions (e.g., `"real_time_limit or (max_evaluations and convergence)"`).

### `HistoryTracker`
Records per-iteration statistics (best, median, worst, full objective, full population, diversity) for post-hoc analysis.

### `Checkpointer`
Saves the full algorithm state to disk at configurable time/iteration intervals using `cloudpickle`.

### `Reporter`
Abstract interface for progress display. Concrete variants: `SilentReporter`, `VerboseReporter`, `TQDMReporter`.

---

## 4. Algorithmic Execution Flow

```
Algorithm.optimize()
  └── Algorithm.initialize()
        └── SearchStrategy.initialize(objfunc)
              └── Initializer.generate_population(objfunc) → Population
              └── Population.calculate_fitness()
  └── loop until StoppingCondition.is_finished():
        └── Algorithm.step()
              ├── SearchStrategy.select_parents(population) → parents
              ├── SearchStrategy.perturb(parents) → offspring
              │     └── Operator.evolve(parents, initializer)
              ├── SearchStrategy.evaluate_population(offspring)
              │     └── Population.calculate_fitness()
              │           └── ObjectiveFunc(population)
              ├── SearchStrategy.repair_population(offspring)
              │     └── ConstraintHandler.repair_solution(matrix)
              ├── SearchStrategy.select_individuals(population, offspring) → new_population
              │     └── SurvivorSelection(population, offspring)
              ├── HistoryTracker.update()
              ├── Checkpointer.step()
              └── Reporter.log_step()
  └── Reporter.log_end()
  └── return best_solution
```

---

## 5. Interfaces and Abstractions

**Strengths:**
- Every component is independently swappable via a clean abstract interface.
- Lambda-based constructors (`ObjectiveFromLambda`, `OperatorFromLambda`, etc.) allow rapid prototyping without subclassing.
- `ParametrizableMixin` enables transparent parameter scheduling across all components without special casing.

**Weaknesses:**
- `SearchStrategy` is not declared `ABC` (no `@abstractmethod`), making it easy to instantiate a non-functional base.
- `Population` is mutable and methods like `take_selection()` return shallow-ish copies. Internal operator functions mutate the genotype matrix in place (e.g., `mutate_noise`, `mutate_sample`), while the call site passes a copy. The contract between mutation and copy semantics is implicit.
- The `ParametrizableMixin` stores kwargs in a plain dict, making it impossible to statically check parameter names at construction time.

---

## 6. Extensibility

| Extension point | Mechanism | Difficulty |
|---|---|---|
| New problem | Subclass `ObjectiveFunc` / `VectorObjectiveFunc` | Low |
| New operator | Subclass `Operator` or use `OperatorFromLambda` | Low |
| New initializer | Subclass `Initializer` or use `InitializerFromLambda` | Low |
| New search strategy | Subclass `SearchStrategy` / `StaticPopulation` | Medium |
| New algorithm | Configure `Algorithm` with custom strategy | Low |
| New stopping criterion | Extend `StoppingCondition` expression grammar | Medium |
| New encoding | Subclass `Encoding` or use `EncodingFromLambda` | Low |
| New constraint | Subclass `RepairConstraint` / `PenalizeConstraint` | Low |
| New schedule | Subclass `SchedulableParameter` | Low |
| New metric | Extend `HistoryTracker` | Medium |

The factory pattern used by `create_operator`, `create_parent_selection`, and `create_survivor_selection` enables string-based configuration but introduces an invisible registry that is hard to introspect or extend without reading factory source files.

---

## 7. Algorithmic Correctness

**Elitism:** `keep_best` and `elitism` survivor selection preserve the historical best. However, no invariant is enforced at the `Algorithm` level; a user can configure a purely generational strategy without elitism and the optimizer may regress.

**Population size invariants:** `StaticPopulation` delegates size consistency to the operator, which may or may not preserve it. There is no assertion that `offspring.pop_size == population.pop_size` after perturbation.

**Fitness caching:** The `Population` has `fitness_calculated` flags, but `recalculate=True` on the objective function bypasses this. Inconsistent use can lead to stale fitness values being used in selection.

**CMA-ES:** Uses `np.random.multivariate_normal` (global NumPy random state) instead of the instance's `self.random_state`, breaking reproducibility guarantees. See `ERRORES.md` → Error 1.

---

## 8. State and Mutability Management

- `Population.genotype_matrix` is a NumPy array shared across operations. Operators that receive `population.genotype_matrix` work on a copy created in `SearchStrategy.perturb()` via `Population.take_selection()`, but this is implicit and fragile.
- Operator functions in `operator_functions/mutation.py` and `operator_functions/crossover.py` mutate their `population` argument in place. Callers must ensure a copy is passed before calling.
- `ParametrizableMixin.store_kwargs()` pops `random_state` from kwargs and stores it, but mutable objects stored in params can be accidentally shared between components.

---

## 9. Randomness Management

- Most components accept `random_state` (int or `np.random.Generator`) and call `check_random_state()` to normalize it.
- `check_random_state(None)` returns `np.random.default_rng()` (unseeded), making experiments with `random_state=None` non-reproducible.
- **Critical bug:** `CMA_ES.initialize()` uses the global `np.random.multivariate_normal` instead of `self.random_state`. Fixed seed has no effect on CMA-ES initialization.
- `sample_distribution` with `ProbDist.VONMISES` falls back to `np.random.uniform` for the default `mu`, also bypassing `random_state`.
- `BranchOperator` and `MaskedOperator` accept `random_state` but the PICK-mode branch does not use a generator.

---

## 10. Testability

**Facilitates testing:**
- Lambda-based constructors make it trivial to build minimal test stubs without full subclasses.
- All major abstractions accept `random_state` for deterministic tests.
- `Population` is a plain data structure with no side effects on construction.
- `StoppingCondition` is a dataclass; its state is directly inspectable.

**Hinders testing:**
- `VerboseReporter` and `TQDMReporter` emit to stdout/tqdm respectively; testing them requires output capture.
- Some internal utilities (`per_individual`, `per_individual_list`) have no direct public API.
- The string-based operator factory is hard to enumerate exhaustively.
- `CMA_ES` global random usage makes reproducibility testing impossible without mocking `np.random`.

---

## 11. Detected Issues

See [`ERRORES.md`](ERRORES.md) for the full issue registry. Key issues:

| # | Module | Severity | Summary |
|---|---|---|---|
| 1 | `strategies/classic/CMA_ES.py` | High | Global `np.random` used instead of `self.random_state` |
| 2 | `operators/operator_functions/mutation.py` | High | Wrong keyword `kappa` passed to `multivariate_normal` in MULTIGAUSS multi-individual path |
| 3 | `operators/operator_functions/mutation.py` | Medium | `ProbDist.VONMISES` default `mu` uses global `np.random.uniform`, bypassing `random_state` |
| 4 | `operators/operator_functions/mutation.py` | Medium | Destructive in-place mutation of the `population` argument without a documented in-place contract |
| 5 | `search_strategy.py` | Low | `SearchStrategy` is not an ABC; accidental instantiation is possible |
| 6 | `stopping_condition.py` | Low | `patience` logic resets on any fitness improvement, but the reset condition depends on floating-point equality |
| 7 | `objective_function.py` | Low | `recalculate=True` silently disables fitness caching without warning |

---

## 12. Objective Evaluation

### Strengths
- **Excellent composability.** The plugin/lambda patterns enable complex algorithms to be built from small, well-tested primitives.
- **Broad algorithm coverage.** GA, DE, ES, CMA-ES, PSO, SA, EDA, BO are all present and share a common interface.
- **Parameter scheduling.** First-class support for adaptive parameters is rare in open-source metaheuristic libraries.
- **Reproducibility mechanisms.** Consistent `random_state` pattern throughout most of the codebase.

### Weaknesses
- **Implicit mutation contracts.** In-place mutation of population arrays is pervasive but undocumented, creating risk for new operator implementations.
- **Random state violations.** Two confirmed cases where the global NumPy state is used instead of the injected generator, undermining reproducibility.
- **Missing abstract enforcement.** `SearchStrategy` lacks `@abstractmethod`, weakening the interface contract.
- **Limited introspection.** The factory registry is not discoverable at runtime; users must read source to know valid string keys.

### Recommendations
1. Enforce `@abstractmethod` on `SearchStrategy` interface methods.
2. Replace all `np.random.*` calls in `CMA_ES` and `mutation.py` with the injected `random_state`.
3. Document or enforce the copy-before-mutate contract for operator functions.
4. Add a runtime check in `Algorithm.step()` that population size is preserved after perturbation.
5. Expose the factory registry as a public dict for introspection.
