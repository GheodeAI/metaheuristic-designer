# ARCHITECTURE.md — Structural Design Critique

**Package:** `metaheuristic-designer` v0.4.0  
**Reviewed:** 2026-05-08  
**Scope:** Source code architecture, interfaces, invariants, extensibility, correctness, and testability. No source code has been modified. All detected defects are documented in `ERRORES.md`.

---

## 1. Package Objective

`metaheuristic-designer` is a Python framework for building, composing, and evaluating **improvement-based metaheuristic algorithms**. Its central paradigm is that every stage of the optimization loop—initialization, evaluation, parent selection, perturbation, constraint handling, survivor selection, stopping, and reporting—is abstracted behind a replaceable interface.

The package supports GA, DE, ES, CMA-ES, PSO, SA, EDA (PBIL, UMDA, CEM), and Bayesian Optimization. A `simple` module provides factory functions for the most common configurations, lowering the entry barrier while preserving full composability for advanced use.

---

## 2. General Repository Structure

```
metaheuristic-designer/
├── src/metaheuristic_designer/
│   ├── algorithm.py              # Top-level orchestrator
│   ├── population.py             # Population data container
│   ├── objective_function.py     # Fitness/objective abstraction
│   ├── initializer.py            # Initializer abstraction
│   ├── operator.py               # Operator abstraction
│   ├── search_strategy.py        # SearchStrategy abstraction (NOT ABC)
│   ├── encoding.py               # Encoding abstraction
│   ├── constraint_handler.py     # Constraint abstraction
│   ├── parent_selection_base.py  # Parent selection abstraction
│   ├── survivor_selection_base.py# Survivor selection abstraction
│   ├── stopping_condition.py     # Stopping criteria (dataclass + expression)
│   ├── history_tracker.py        # Per-iteration statistics
│   ├── checkpointer.py           # State persistence (cloudpickle)
│   ├── reporter.py               # Progress reporting abstraction
│   ├── parametrizable_mixin.py   # Parameter scheduling mixin
│   ├── schedulable_parameter.py  # Schedulable parameter abstraction
│   ├── utils.py                  # Helpers (check_random_state, JSON encoder)
│   ├── algorithms/               # Compound algorithm wrappers (VNS, etc.)
│   ├── benchmarks/               # Benchmark functions (CEC, sphere, classic problems)
│   ├── constraint_handlers/      # Clip, bounce, cycle, linear-penalty, composite
│   ├── encodings/                # TypeCast, Sigmoid, Composite, PSO, SelfAdapting
│   ├── initializers/             # Uniform, Gaussian, Perm, Seed, Direct, Extended
│   ├── operators/                # Operator factories + operator_functions/
│   ├── parameter_schedules/      # Linear, exponential, logistic, random, threshold, step
│   ├── parent_selection/         # Tournament, roulette, SUS, rank, etc.
│   ├── reporters/                # Silent, verbose, tqdm
│   ├── simple/                   # Convenience constructors (hill_climb_real, etc.)
│   ├── strategies/               # GA, DE, ES, CMA-ES, SA, PSO, EDA, BO, local search
│   └── survivor_selection/       # Keep-best, generational, (μ+λ), (μ,λ), etc.
├── tests/                        # pytest + hypothesis test suite
├── docs/                         # Sphinx documentation source
├── examples/                     # Usage scripts
├── data/                         # TSP and SAT instance files
└── pyproject.toml                # Metadata, coverage, linting config
```

**Critique:** The flat `src/metaheuristic_designer/` layout mixes abstract base modules (e.g., `operator.py`, `encoding.py`) with top-level orchestration (`algorithm.py`, `stopping_condition.py`). This makes it difficult for newcomers to distinguish interface definitions from concrete utilities. A subdirectory `core/` or `base/` would clarify what is meant to be subclassed vs. what is machinery.

---

## 3. Domain Model

### `Population`

The central data container. Holds:
- `genotype_matrix` (`n_individuals × dimension`, NumPy array)
- `fitness` and `objective` (1-D arrays, mutable in place)
- `best_solution` tracking (updated via `update_best_from_parents`)
- Optional `Encoding` for phenotype decoding

**Critique:** `Population` is mutable. Several methods (`update_fitness`, `take_selection`, `sort_population`) return new objects, while others mutate `self` in place (`calculate_fitness`, `repeat`). The dual mutation/copy model is not documented and forces callers to reason about object identity rather than relying on explicit contracts. The `best_solution` tracking state embedded in `Population` creates implicit coupling with the `Algorithm` loop.

### `ObjectiveFunc`

Abstract base. Subclasses implement `objective(solution)`. The base class handles constraint penalties, vectorization, and caching (`recalculate` flag).

**Critique:** The `recalculate` flag exists on the objective function itself, not on the algorithm or population. This placement is architecturally wrong: the decision of when to recompute fitness belongs to the optimization loop (`Algorithm.step`), not the problem definition. A side-effect-free `ObjectiveFunc` would never carry this flag.

### `Algorithm`

Top-level orchestrator. Manages the optimization loop, stopping conditions, progress reporting, history tracking, and checkpointing. Delegates search logic entirely to `SearchStrategy`.

**Critique:** `Algorithm` is well-factored. However, it does not enforce or check that the population size is consistent across iterations. A `SearchStrategy` that silently changes `pop_size` will not be caught.

### `SearchStrategy`

Defines the template method pattern for the algorithm lifecycle: initialize → perturb → repair → select.

**Critique:** `SearchStrategy` does **not** inherit from `ABC` and has no `@abstractmethod` decorators. Methods like `perturb()` raise `NotImplementedError` at runtime instead of being declared abstract. This means:
- An incomplete subclass instantiates silently.
- Static analyzers and IDEs cannot warn about missing overrides.
- `SearchStrategy` itself can be instantiated, which is semantically wrong.

### `Operator`

Transforms a population into offspring. Abstract method: `evolve(population, initializer) → Population`.

**Critique:** `CompositeOperator.evolve()` calls `op.evolve(population, ...)` directly on sub-operators, bypassing `op.__call__()`. This skips encoding transformations declared on each sub-operator. A pipeline of operators with heterogeneous encodings inside a `CompositeOperator` will silently ignore those encodings (see `ERRORES.md` → Error 10).

### `Initializer`

Generates an initial population. Variants cover uniform, Gaussian, permutation, seeded, direct-injection, and extended initializers.

**Critique:** `SeedProbInitializer.generate_individual()` does not handle `Population` objects as `solutions`, while the sibling class `SeedDetermInitializer` does. This asymmetry creates a confusing and unreliable API. See `ERRORES.md` → Error 15.

### `Encoding`

Maps genotype → phenotype. Two distinct sub-hierarchies exist:
1. `Encoding` → `DefaultEncoding`, `TypeCastEncoding`, `SigmoidEncoding`
2. `ParameterExtendingEncoding` (subclass of `Encoding`) → `CompositeEncoding`, `PSO_encoding`, `self_adapting_ES_encoding`

**Critique:** `CompositeEncoding` calls `encoding.encode_func()` and `encoding.decode_func()` on its members. These methods only exist on `ParameterExtendingEncoding`, not on the base `Encoding`. There is no runtime check that validates members are the correct subtype. Passing a `TypeCastEncoding` or `SigmoidEncoding` as a member of `CompositeEncoding` causes an `AttributeError` that is invisible until evolution begins. See `ERRORES.md` → Error 13.

### `ConstraintHandler`

Two sub-hierarchies: `RepairConstraint` (repair + no penalty) and `PenalizeConstraint` (penalty + no-op repair). `CompositeConstraint` chains multiple handlers.

**Critique:** Well-designed. The repair-vs-penalty split is clean, and `CompositeConstraint` correctly sums penalties and chains repairs in order.

### `ParentSelection` / `SurvivorSelection`

Abstract selection interfaces. Concrete implementations include tournament, roulette, SUS, elitism, rank, generational, (μ+λ), and (μ,λ) selection.

**Critique:** Both hierarchies expose a `from_name()` factory pattern. The valid string keys are not documented as constants or as a public registry, requiring users to read source to determine valid values.

### `SchedulableParameter` / `ParametrizableMixin`

Any numeric parameter can be a `SchedulableParameter` that evolves over time. `ParametrizableMixin` is inherited by all major abstractions.

**Critique:** `ExponentialDecaySchedule` updates `curr_value` in place on each `evaluate()` call but has no `restart()` override. Multi-restart experiments will resume from a stale decayed value. See `ERRORES.md` → Error 8. Furthermore, `ParametrizableMixin.store_kwargs()` pops `random_state` from kwargs and stores it; this means passing `random_state` in kwargs has a side effect that is not obvious from the call site.

### `StoppingCondition`

Dataclass with fields for all criteria (`max_iterations`, `max_evaluations`, `real_time_limit`, `cpu_time_limit`, `objective_target`, `max_patience`). Supports compound boolean expressions via a string grammar (e.g., `"max_evaluations and not objective_target"`).

**Critique:** The string expression grammar is powerful but not validated at construction time. A typo like `"max_iteration"` (missing 's') is silently ignored rather than raising a `ValueError`. This makes misconfigured stopping criteria hard to debug.

### `HistoryTracker`

Records per-iteration statistics. Attached to `Algorithm` via constructor parameter `history_tracker=`.

**Critique:** The tracker must be injected at construction time; there is no `Algorithm.set_tracker()` method. If a user forgets to inject it before `optimize()`, no history is available and no warning is given.

---

## 4. Algorithmic Execution Flow

```
Algorithm.optimize()
  └── Algorithm.initialize()
        └── SearchStrategy.initialize(objfunc)
              ├── Initializer.generate_population(objfunc) → Population
              └── Population.calculate_fitness()
  └── loop until StoppingCondition.is_finished():
        └── Algorithm.step()
              ├── SearchStrategy.select_parents(population) → parents
              ├── SearchStrategy.perturb(parents) → offspring
              │     └── Operator.evolve(parents, initializer)
              │           └── operator_functions.*(genotype_matrix)  ← mutates in-place
              ├── SearchStrategy.evaluate_population(offspring)
              │     └── Population.calculate_fitness()
              │           └── ObjectiveFunc.__call__(population)
              ├── SearchStrategy.repair_population(offspring)
              │     └── ConstraintHandler.repair_solution(matrix)
              ├── SearchStrategy.select_individuals(population, offspring) → new_population
              │     └── SurvivorSelection(population, offspring)
              ├── HistoryTracker.update()
              ├── Checkpointer.step()
              └── Reporter.log_step()
  └── Reporter.log_end()
  └── return population (caller retrieves .best_solution())
```

**Critique:** The `operator_functions.*` layer receives and mutates NumPy arrays directly. The contract that the caller must pass a copy before calling these functions is nowhere enforced or documented. Adding a new operator that passes `population.genotype_matrix` without copying will silently corrupt parent population fitness values that have already been evaluated and cached.

---

## 5. Interfaces and Abstractions

### What works well

- Every major component is independently swappable.
- Lambda constructors (`ObjectiveFromLambda`, `OperatorFromLambda`, `ConstraintHandlerFromLambda`, etc.) allow rapid prototyping without subclassing.
- `ParametrizableMixin` enables transparent parameter scheduling across all components without case-specific handling.
- `check_random_state()` normalizes seeds consistently throughout most of the codebase.

### What is broken or inconsistent

| Issue | Location | Severity |
|---|---|---|
| `SearchStrategy` is not `ABC`; `@abstractmethod` missing | `search_strategy.py` | High |
| `CompositeOperator` bypasses sub-operator encodings | `operators/composite_operator.py` | High |
| `CompositeEncoding` does not validate member types | `encodings/composite_encoding.py` | High |
| `SeedProbInitializer` asymmetry with `SeedDetermInitializer` | `initializers/seed_initializer.py` | Medium |
| `ExponentialDecaySchedule` lacks `restart()` | `parameter_schedules/exponential_decay_schedule.py` | Medium |
| `StoppingCondition` expression grammar not validated at construction | `stopping_condition.py` | Low |
| `HistoryTracker` not attachable post-construction | `algorithm.py` | Low |

---

## 6. Extensibility

| Extension point | Mechanism | Difficulty | Notes |
|---|---|---|---|
| New problem | Subclass `ObjectiveFunc` / `VectorObjectiveFunc` | Low | Clean interface |
| New operator | Subclass `Operator` or `OperatorFromLambda` | Low | `evolve()` is the only required method |
| New initializer | Subclass `Initializer` or `InitializerFromLambda` | Low | Clean interface |
| New search strategy | Subclass `SearchStrategy` / `StaticPopulation` | Medium | Not `ABC`; mistakes silently accepted |
| New algorithm | Configure `Algorithm` with custom strategy | Low | |
| New stopping criterion | Extend `StoppingCondition` expression grammar | Medium | Grammar not validated; typos silently ignored |
| New encoding | Subclass `Encoding` or `EncodingFromLambda` | Low | Risk: `CompositeEncoding` may silently break |
| New constraint | Subclass `RepairConstraint` / `PenalizeConstraint` | Low | |
| New schedule | Subclass `SchedulableParameter` | Low | Must also add `restart()` to avoid drift |
| New metric | Extend `HistoryTracker` | Medium | Must be injected at construction; no plugin API |

**String-based factory registries** (`create_operator`, `create_parent_selection`, etc.) introduce implicit "magic string" configuration that is not discoverable without reading source. A public `dict` or `enum` of valid keys would solve this.

---

## 7. Algorithmic Correctness

**Elitism:** `keep_best` and `elitism` survivor selection preserve the historical best solution. There is no invariant enforced at the `Algorithm` level; a purely generational strategy can cause fitness regression, and this is by design, but it is not documented at the `Algorithm` API level.

**Population size invariants:** `StaticPopulation` does not check that `offspring.pop_size == population.pop_size` after `perturb()`. A bug in a custom operator that changes population size will propagate silently until it causes a shape mismatch in a later step.

**Fitness caching:** `Population.fitness_calculated` flags exist, but `recalculate=True` on `ObjectiveFunc` bypasses them globally. The combination of cached flags and an unconditional-recalculate objective creates potential for stale fitness data to be used in selection if the flag is partially updated.

**CMA-ES (broken):** Uses `np.random.multivariate_normal` (global state) instead of `self.random_state`, breaking reproducibility. Additionally, `self.offspring_size` does not exist as a direct attribute; the value is stored as `self.params.offspring_size`, causing `AttributeError` on `initialize()`. CMA-ES is effectively non-functional in the current codebase. See `ERRORES.md` → Errors 1 and 11.

**MULTIGAUSS multi-individual path (broken):** The per-individual branch of `sample_distribution` passes `kappa=` to `scipy.stats.multivariate_normal`, which does not have a `kappa` parameter (it belongs to `vonmises_fisher`). This raises `TypeError` at runtime for any use of per-individual covariance matrices. See `ERRORES.md` → Error 2.

**EDA reproducibility:** Confirmed working. `BernoulliPBIL`, `BernoulliUMDA`, `GaussianPBIL`, `GaussianUMDA`, and `CrossEntropyMethod` all rely on the injected `self.random_state` for sampling, and produce identical results across independent runs when given the same seed.

---

## 8. State and Mutability Management

### Explicit copies

`SearchStrategy.perturb()` creates offspring via `Population.take_selection()` before passing to operators, providing some protection. However:

- `Population.take_selection()` creates a new `Population` object with a copy of the selected rows, but the calling pattern is operator-specific.
- `CompositeOperator` passes the same `population` object through successive `op.evolve()` calls; each operator mutates what the previous left behind.

### Implicit in-place mutation in operator functions

All functions in `operators/operator_functions/mutation.py` accept a `population` NumPy array and mutate it in place:

```python
# mutate_noise (line 170):
population[mask_pos] = population[mask_pos] + ...
return population  # same object
```

The function returns the same object it received, making the behavior look like a pure function to callers that ignore the return value. This is a latent hazard for any new operator author who is not aware of the convention.

### `ParametrizableMixin` shared state

`ParametrizableMixin.store_kwargs()` pops `random_state` from kwargs and stores it as a separate instance attribute. If the same `random_state` generator object is passed to two components (which is possible when using `random_state=rng` directly), both components advance the same RNG, creating unexpected coupling between operator trajectories.

---

## 9. Randomness Management

The package follows a consistent `random_state` pattern: every component accepts it, and `check_random_state()` in `utils.py` normalizes seeds, integers, and generator objects to `numpy.random.Generator`. This is a strong design choice that supports reproducibility in the majority of cases.

**Confirmed violations:**

| Location | Problem |
|---|---|
| `strategies/classic/CMA_ES.py` line 82 | `np.random.multivariate_normal` (global state) |
| `operators/operator_functions/mutation.py` line 303 | `np.random.uniform` for VONMISES default `mu` |

**Structural weakness:** `check_random_state(None)` returns an unseeded `np.random.default_rng()`. A user who omits `random_state` gets a non-reproducible run with no warning. A `random_state` default of `0` (or any fixed integer) would be safer for research reproducibility.

---

## 10. Testability

### Facilitates testing

- Lambda constructors enable minimal test stubs for every extension point.
- `random_state` acceptance across all components allows deterministic unit and integration tests.
- `Population` is a plain data structure; constructing test populations is straightforward.
- `StoppingCondition` is a dataclass; its state is directly inspectable.
- `SilentReporter` exists and is used throughout tests.

### Hinders testing

| Barrier | Impact |
|---|---|
| `SearchStrategy` not `ABC` | No guarantee that a stub subclass is correct |
| In-place operator mutations undocumented | Tests must know to copy before comparing |
| `HistoryTracker` must be injected at construction | Easy to write tests that have no history and never notice |
| `VerboseReporter`/`TQDMReporter` write to stdout/tqdm | Testing requires output capture or patching |
| String-based factory registries | Exhaustive operator testing requires reading source for valid keys |
| CMA-ES global RNG | Reproducibility tests for CMA-ES impossible without mocking `np.random` |

---

## 11. Detected Issues Summary

Full details in [`ERRORES.md`](ERRORES.md).

| # | Module | Severity | Summary |
|---|---|---|---|
| 1 | `strategies/classic/CMA_ES.py` | **Critical** | Global `np.random` used instead of `self.random_state` |
| 2 | `operators/operator_functions/mutation.py` | **Critical** | Wrong kwarg `kappa` to `multivariate_normal`; always raises `TypeError` |
| 3 | `operators/operator_functions/mutation.py` | **High** | VONMISES default `mu` uses global `np.random.uniform` |
| 4 | `operators/operator_functions/mutation.py` | **High** | Undocumented in-place mutation of `population` argument |
| 5 | `search_strategy.py` | **High** | `SearchStrategy` not `ABC`; incomplete subclasses accepted silently |
| 6 | `stopping_condition.py` | Medium | Patience comparison not epsilon-guarded; floating-point fragility |
| 7 | `objective_function.py` | Medium | `recalculate=True` silently disables caching without warning |
| 8 | `parameter_schedules/exponential_decay_schedule.py` | Medium | No `restart()` override; decayed value persists across restarts |
| 9 | `operators/branch_operator.py` | Low | PICK-mode branch contract undocumented |
| 10 | `operators/composite_operator.py` | **High** | Sub-operator encodings bypassed; silent correctness failure |
| 11 | `strategies/classic/CMA_ES.py` | **Critical** | `self.offspring_size` does not exist; CMA-ES non-functional |
| 12 | `operators/operator_functions/mutation.py` | **Critical** | `multivariate_categorical.rvs()` shape mismatch for non-matching sizes |
| 13 | `encodings/composite_encoding.py` | **High** | No type validation of members; `AttributeError` at runtime for non-PE encodings |
| 14 | `population.py` | Medium | `debug_repr()` raises `ValueError` on empty population |
| 15 | `initializers/seed_initializer.py` | Medium | `SeedProbInitializer` fails with `Population` solutions; inconsistent with sibling class |
| 16 | `strategies/EDA/PBIL.py` | None | EDA reproducibility confirmed (no bug; earlier `xfail` removed) |

---

## 12. Objective Evaluation

### Strengths

1. **Composability.** The plugin/lambda pattern is consistently applied across all extension points. Complex algorithms can be assembled from small, independently verifiable primitives without subclassing.

2. **Algorithm breadth.** GA, DE, ES, PSO, SA, EDA (PBIL, UMDA, CEM), and Bayesian Optimization share a single lifecycle interface. Adding a new algorithm family requires only subclassing `SearchStrategy`.

3. **First-class parameter scheduling.** `SchedulableParameter` and `ParametrizableMixin` give adaptive parameters the same status as fixed parameters. This is rare in open-source metaheuristic libraries and enables realistic adaptive algorithm implementations.

4. **Reproducibility infrastructure.** `check_random_state()` and consistent `random_state` injection are applied across most components. When used correctly, the framework supports fully reproducible experiments.

5. **`simple` module.** The factory functions lower the entry barrier significantly without hiding the composable API for advanced users.

### Weaknesses

1. **Broken critical paths.** CMA-ES is non-functional due to two independent bugs (Errors 1 and 11). The MULTIGAUSS multi-individual path is broken (Error 2). These are not edge cases; they are advertised features.

2. **Implicit mutation semantics.** The convention that operator functions mutate their argument in place is universally applied but never documented. New developers who follow the pattern of "create a subclass of `Operator`" and call an existing `operator_functions.*` helper will inadvertently corrupt parent populations.

3. **Missing abstract method enforcement.** `SearchStrategy` without `@abstractmethod` is the single most impactful design weakness. It inverts the fail-fast principle: bugs appear at execution time inside the loop rather than at instantiation time.

4. **Inconsistent sibling implementations.** `SeedProbInitializer` vs. `SeedDetermInitializer` (Error 15), `CompositeEncoding` members with `encode_func`/`decode_func` (Error 13), and `CompositeOperator` encoding bypass (Error 10) are three independent cases where two closely related components handle the same scenario differently. This suggests insufficient code review or missing integration tests.

5. **Random state violations in high-visibility components.** The global `np.random` usage in CMA-ES and VONMISES sampling undermines the otherwise solid reproducibility story.

### Recommendations

1. **Declare `SearchStrategy` as `ABC`** and mark all strategy-specific methods with `@abstractmethod`. This is a one-line change per method that eliminates an entire class of silent failures.

2. **Fix CMA-ES** (Errors 1 and 11) before including it in any publication or benchmark. Replace `np.random.multivariate_normal` with `self.random_state.multivariate_normal` and replace `self.offspring_size` with `self.params.offspring_size`.

3. **Document the in-place mutation contract** with an explicit warning in `operator_functions/mutation.py` and in the `Operator` subclassing guide.

4. **Add a type check in `CompositeEncoding.__init__`** that validates all members are `ParameterExtendingEncoding` instances and raises a descriptive `TypeError` early.

5. **Add a `restart()` override to `ExponentialDecaySchedule`** that resets `self.curr_value = self.init_value`.

6. **Validate `StoppingCondition` expressions at construction time** using the existing expression parser rather than deferring to execution.

7. **Make the factory registries public constants** so users can enumerate valid operator, selection, and initializer strings without reading source code.
