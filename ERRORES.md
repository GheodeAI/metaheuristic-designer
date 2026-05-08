# ERRORES.md

This document records all issues detected during static analysis, source code inspection, and test development for `metaheuristic-designer` v0.4.0. No source code has been modified; issues are described only. Corrections are conceptual.

---

## Error 1: CMA-ES uses global NumPy RNG instead of injected `random_state`

- **File/script:** `src/metaheuristic_designer/strategies/classic/CMA_ES.py`
- **Affected element:** `CMA_ES.initialize()`, line 82
- **Description:** The method calls `np.random.multivariate_normal(mean=mean, cov=cov_matrix, size=self.offspring_size)` using the global NumPy random state. The class accepts and stores `random_state` in `self.random_state` (line 35), which is never used for actual sampling.
- **Impact:** Running `Algorithm` with a fixed seed does not produce reproducible results when using CMA-ES. Benchmark comparisons and regression tests against CMA-ES are unreliable.
- **Evidence:** `CMA_ES.__init__` calls `check_random_state(random_state)` and stores the result in `self.random_state`, but `initialize()` bypasses it entirely.
- **Possible correction:** Replace `np.random.multivariate_normal(...)` with `self.random_state.multivariate_normal(mean=mean, cov=cov_matrix, size=self.offspring_size)`.

---

## Error 2: Wrong keyword argument `kappa` passed to `scipy.stats.multivariate_normal` in MULTIGAUSS multi-individual path

- **File/script:** `src/metaheuristic_designer/operators/operator_functions/mutation.py`
- **Affected element:** `sample_distribution()`, line 279–280 (MULTIGAUSS multi-individual branch)
- **Description:** In the branch that handles a different mean/covariance per individual (`mean.ndim > 1 or cov.ndim > 2`), the code constructs `sp.stats.multivariate_normal(mean=mean_i, kappa=scale_i)`. The `kappa` parameter does not exist in `scipy.stats.multivariate_normal`; it belongs to `scipy.stats.vonmises_fisher`. The correct parameter is `cov`.
- **Impact:** Any use of per-individual MULTIGAUSS sampling (e.g., CMA-ES with per-individual covariance matrices) will raise a `TypeError` at runtime.
- **Evidence:** `scipy.stats.multivariate_normal` signature is `(mean=None, cov=1, allow_singular=False, ...)`. The keyword `kappa` is rejected.
- **Possible correction:** Replace `kappa=scale_i` with `cov=scale_i`.

---

## Error 3: `ProbDist.VONMISES` default `mu` uses global `np.random.uniform`, bypassing `random_state`

- **File/script:** `src/metaheuristic_designer/operators/operator_functions/mutation.py`
- **Affected element:** `sample_distribution()`, line 303
- **Description:** When `ProbDist.VONMISES` is used and no `mu` is provided in kwargs, the default is computed as `np.random.uniform(-1, 1, shape)` using the global NumPy random state, not the `random_state` parameter that was passed to `sample_distribution`.
- **Impact:** VONMISES sampling with default `mu` is non-reproducible even when a fixed seed is provided. This undermines experimental reproducibility for operators using the von Mises-Fisher distribution.
- **Evidence:** `sample_distribution` signature receives `random_state` and passes it to `prob_distrib.rvs(...)`, but the `mu` default construction on line 303 uses `np.random.uniform`.
- **Possible correction:** Replace `np.random.uniform(-1, 1, shape)` with `random_state.uniform(-1, 1, shape)`.

---

## Error 4: Operator functions mutate the `population` argument in place without a documented contract

- **File/script:** `src/metaheuristic_designer/operators/operator_functions/mutation.py`
- **Affected element:** `mutate_sample()`, `mutate_noise()`, `xor_mask()`
- **Description:** These functions accept a `population` NumPy array and modify it in place (e.g., `population[mask_pos] = ...`), then return the same object. The docstrings do not state that the input is mutated. Callers in the upper layers pass a copy, so the current behavior is safe in practice, but the contract is implicit and fragile.
- **Impact:** A new operator implementation that calls these functions without copying first will accidentally mutate the parent population, corrupting fitness values and history tracking. This is a latent extensibility hazard.
- **Evidence:** `mutate_noise()` lines 170: `population[mask_pos] = population[mask_pos] + ...`. The function returns `population` (same object). No copy is made inside.
- **Possible correction:** Either document the in-place contract explicitly in the docstring, or make a copy at the start of each function (`population = population.copy()`) to make the functions pure.

---

## Error 5: `SearchStrategy` is not declared as an Abstract Base Class

- **File/script:** `src/metaheuristic_designer/search_strategy.py`
- **Affected element:** `SearchStrategy` class definition
- **Description:** `SearchStrategy` inherits from `ParametrizableMixin` only. It has no `@abstractmethod` decorators. Methods like `perturb()` have implementations that raise `NotImplementedError`, but this is only discoverable at runtime.
- **Impact:** A subclass that forgets to override `perturb()` will instantiate successfully and fail only when `step()` is called. IDE type checkers and static analyzers cannot warn about incomplete implementations. Additionally, `SearchStrategy` itself can be instantiated, which is semantically wrong.
- **Evidence:** `SearchStrategy.perturb()` raises `NotImplementedError` on line 179 (runtime guard only). No `ABC` in the class hierarchy.
- **Possible correction:** Add `from abc import ABC, abstractmethod` and mark `perturb()` (and other strategy-specific methods) as `@abstractmethod`. Inherit from `ABC`.

---

## Error 6: `StoppingCondition.patience` convergence check is sensitive to floating-point equality

- **File/script:** `src/metaheuristic_designer/stopping_condition.py`
- **Affected element:** `StoppingCondition.step()` method, patience logic
- **Description:** The patience counter resets when the best fitness improves. Improvement detection compares new and stored best fitness values. For objectives with floating-point outputs, tiny numerical noise can prevent the counter from resetting (false stagnation) or, conversely, reset it on negligible improvements (infinite patience).
- **Impact:** Early stopping based on patience may fire prematurely or never fire, depending on the problem's fitness landscape and numerical precision. This makes convergence behavior unpredictable.
- **Evidence:** A tolerance-free `>` or `<` comparison is used in the patience update path without any epsilon guard.
- **Possible correction:** Add a configurable `patience_delta` threshold. Reset patience only if `|new_best - old_best| > patience_delta`.

---

## Error 7: `ObjectiveFunc` with `recalculate=True` silently disables fitness caching

- **File/script:** `src/metaheuristic_designer/objective_function.py`
- **Affected element:** `ObjectiveFunc.__call__()` / `fitness()`, the `recalculate` flag
- **Description:** When `recalculate=True`, every call to `__call__` or `fitness` recomputes the objective function for all individuals, even those already evaluated. There is no warning or log message indicating that caching has been disabled. Users who accidentally pass `recalculate=True` will silently incur O(pop_size) redundant evaluations per step.
- **Impact:** Significant performance regression in expensive objective functions. Hard to diagnose without profiling.
- **Evidence:** `recalculate` flag is checked in `fitness()` to decide whether to skip cached values; no `logging.warning` call accompanies it.
- **Possible correction:** Add a `logging.warning` or `logging.debug` message when `recalculate=True` is active, and document the performance trade-off in the docstring.

---

## Error 8: `ExponentialDecaySchedule` does not reset `curr_value` on `restart()`

- **File/script:** `src/metaheuristic_designer/parameter_schedules/exponential_decay_schedule.py`
- **Affected element:** `ExponentialDecaySchedule` class, missing `restart()` override
- **Description:** In iterative mode, `ExponentialDecaySchedule` updates `self.curr_value` on each call to `evaluate()`. If the algorithm is restarted via `Algorithm.restart()`, other components are reset but `ExponentialDecaySchedule.curr_value` remains at its decayed value from the previous run, causing the schedule to continue from an arbitrary intermediate point.
- **Impact:** Algorithm restarts with `ExponentialDecaySchedule` will produce incorrect parameter trajectories. Reproducibility of multi-restart experiments is broken.
- **Evidence:** `SchedulableParameter` (base class) does not define a `restart()` method. `ExponentialDecaySchedule.__init__` sets `self.curr_value = init_value`, but no mechanism resets this after a restart.
- **Possible correction:** Add a `restart()` method that sets `self.curr_value = self.init_value`.

---

## Error 9: `BranchOperator` PICK mode does not use the instance `random_state`

- **File/script:** `src/metaheuristic_designer/operators/branch_operator.py`
- **Affected element:** `BranchOperator.choose_index()`, PICK mode branch
- **Description:** The RANDOM mode uses `self.random_state.choice(...)` for reproducible operator selection. The PICK mode, however, selects operators by deterministic index and does not use any random state. If a user expects PICK to cycle deterministically, this is correct behavior; but the lack of a documented contract makes the intent unclear.
- **Impact:** Minor: users expecting RANDOM behavior might configure PICK by mistake. The absence of documentation may cause confusion.
- **Evidence:** `choose_index()` has two code paths: RANDOM (uses `self.random_state`) and PICK (index-based, no RNG). The class docstring does not explain the distinction.
- **Possible correction:** Document the RANDOM vs PICK contract in the class docstring and parameter descriptions.

---

## Error 11: `CMA_ES.initialize()` uses `self.offspring_size` which does not exist as a direct attribute

- **File/script:** `src/metaheuristic_designer/strategies/classic/CMA_ES.py`
- **Affected element:** `CMA_ES.initialize()`, line 82
- **Description:** `CMA_ES.initialize()` calls `np.random.multivariate_normal(..., size=self.offspring_size)`. However, `VariablePopulation` stores `offspring_size` via `self.update_kwargs(offspring_size=...)`, making it accessible only as `self.params.offspring_size`, not as `self.offspring_size`.
- **Impact:** `CMA_ES.initialize()` will always raise `AttributeError: 'CMA_ES' object has no attribute 'offspring_size'` when called. The CMA-ES algorithm is effectively non-functional.
- **Evidence:** `AttributeError: 'CMA_ES' object has no attribute 'offspring_size'` raised during `strategy.initialize(objfunc)` in all test attempts.
- **Possible correction:** Replace `self.offspring_size` with `self.params.offspring_size`.

---

## Error 12: `multivariate_categorical.rvs()` produces shape mismatch when called with `size` as scalar

- **File/script:** `src/metaheuristic_designer/operators/operator_functions/mutation.py`
- **Affected element:** `multivariate_categorical.rvs()`, lines 84–95
- **Description:** When `rvs(size=n)` is called with a scalar `n`, the implementation computes `size = (n, len(categories))` and generates `index_rnd` of shape `(n, len(categories))`. However, `self.cumsum_matrix` has shape `(n_rows, len(categories))` with batch shape `(n_rows,)`. When `n != n_rows`, the numpy `vectorize` call fails with a shape mismatch error when broadcasting `(n_rows,)` against `(n, len(categories))`.
- **Impact:** `MULTICATEGORICAL` distribution sampling via `sample_distribution()` fails whenever the requested `size` differs from the number of rows in the weight matrix. This makes the operator non-functional for most practical configurations.
- **Evidence:** `ValueError: shape mismatch: objects cannot be broadcast to a single shape` when calling `sample_distribution((2, 3), distrib=ProbDist.MULTICATEGORICAL, p=weights_matrix)`.
- **Possible correction:** Rethink `rvs()` to either use vectorized indexing without `np.vectorize`, or generate `index_rnd` as a 1D array of shape `(size,)` and use `self.cumsum_matrix` row lookup individually.

---

## Error 10: `CompositeOperator` silently ignores encodings of sub-operators

- **File/script:** `src/metaheuristic_designer/operators/composite_operator.py`
- **Affected element:** `CompositeOperator.evolve()`
- **Description:** `CompositeOperator` applies a list of operators sequentially. Each sub-operator may have its own encoding, which decodes/encodes the population before/after evolution. The `CompositeOperator` calls `op.evolve(population, ...)` directly without chaining the encoding transformations. Sub-operator encodings that rely on the population being in a specific encoded form may behave incorrectly.
- **Impact:** Pipelines that combine operators with heterogeneous encodings inside a `CompositeOperator` may silently produce wrong results.
- **Evidence:** `CompositeOperator.evolve()` iterates `for op in self.operators: population = op.evolve(population, initializer)`. The base `Operator.__call__` applies encoding transformations, but `evolve` is called directly, bypassing them.
- **Possible correction:** Call `op(population, initializer)` (i.e., `__call__`) instead of `op.evolve(population, initializer)` to trigger encoding transformations.

---

## Error 13: `CompositeEncoding.encode_func` / `decode_func` crash on non-`ParameterExtendingEncoding` members

- **File/script:** `src/metaheuristic_designer/encodings/composite_encoding.py`
- **Affected element:** `CompositeEncoding.encode_func()`, `CompositeEncoding.decode_func()`
- **Description:** `CompositeEncoding` is designed for `ParameterExtendingEncoding` sub-encodings. It iterates over `self.encodings` and calls `encoding.encode_func(population)` / `encoding.decode_func(population)`. However, the base `Encoding` class only defines `encode(population)` and `decode(population)`, not `encode_func` and `decode_func`. If any member of `self.encodings` is a plain `Encoding` subclass (not a `ParameterExtendingEncoding`), an `AttributeError` is raised.
- **Impact:** Using `CompositeEncoding` with non-`ParameterExtendingEncoding` members causes `AttributeError` at runtime. The failure is silent until the encoding is actually used during evolution.
- **Evidence:** `AttributeError: 'TypeCastEncoding' object has no attribute 'encode_func'` raised when calling `composite.decode_func(pop)` with a `TypeCastEncoding` member.
- **Possible correction:** Add a runtime check in `CompositeEncoding.__init__` that validates all members are `ParameterExtendingEncoding` instances, and raise a descriptive `TypeError` early. Alternatively, fall back to `encode`/`decode` for members that do not have `encode_func`/`decode_func`.

---

## Error 14: `Population.debug_repr()` raises `ValueError` on an empty population

- **File/script:** `src/metaheuristic_designer/population.py`
- **Affected element:** `Population.debug_repr()`
- **Description:** `debug_repr()` calls `self.fitness.min()` and `self.fitness.max()` unconditionally. When the population has zero individuals (empty `genotype_matrix`), `fitness` is an empty array and `min()`/`max()` raise `ValueError: zero-size array to reduction operation minimum which has no identity`.
- **Impact:** Any diagnostic code that calls `debug_repr()` on an empty or uninitialized population (e.g., during debugging, testing, or logging) will crash with an uninformative `ValueError`.
- **Evidence:** `ValueError: zero-size array to reduction operation minimum which has no identity` when constructing `Population(objfunc, np.empty((0, 4)))` and calling `.debug_repr()`.
- **Possible correction:** Add an early guard: if `len(self.fitness) == 0`, return a placeholder string like `"Population(empty)"`.

---

## Error 15: `SeedProbInitializer.generate_individual()` fails when `solutions` is a `Population` object

- **File/script:** `src/metaheuristic_designer/initializers/seed_initializer.py`
- **Affected element:** `SeedProbInitializer.generate_individual()`
- **Description:** When `solutions` is a `Population` object, `generate_individual()` calls `self.random_state.choice(self.solutions, axis=0)`. NumPy's `random_state.choice` does not accept a `Population` object; it requires a 1-D array or an integer. This raises `ValueError: a must be a 1-dimensional array or an integer`. In contrast, `SeedDetermInitializer` correctly handles the `Population` case by accessing `.genotype_matrix`.
- **Impact:** `SeedProbInitializer` cannot be used with `Population` objects as `solutions`, despite `SeedDetermInitializer` supporting this use case. The inconsistency in the sibling class creates a confusing API.
- **Evidence:** `ValueError: a must be a 1-dimensional array or an integer` when calling `init.generate_individual()` with a `Population` passed as `solutions`.
- **Possible correction:** In `SeedProbInitializer.generate_individual()`, add a check similar to `SeedDetermInitializer`: `if isinstance(self.solutions, Population): solutions_matrix = self.solutions.genotype_matrix else: solutions_matrix = self.solutions`. Then sample from `solutions_matrix`.

---

## Error 16: EDA strategies are not fully reproducible across independent runs with the same seed

- **File/script:** `src/metaheuristic_designer/strategies/EDA/PBIL.py` and/or `src/metaheuristic_designer/initializers/`
- **Affected element:** `BernoulliPBIL` (and likely all EDA strategies)
- **Description:** Two consecutive calls to `_build_pbil(dim=8, seed=42).optimize().best_solution()[1]` with identical configuration produce different fitness values (observed: 8.0 vs 7.0). Despite seeding both the strategy and the initializer with the same value, something advances a global or shared random state between runs. This is consistent with a component somewhere in the EDA pipeline calling `np.random.*` (global state) instead of `self.random_state`.
- **Impact:** EDA experiments are not reproducible. Running the same algorithm twice with the same seed will produce different results, making benchmark comparisons and regression testing unreliable.
- **Evidence:** `test_pbil_binary_reproducible` fails when run after other tests in the suite, indicating that global RNG state is being consumed and not reset between runs. The test is marked `xfail`.
- **Possible correction:** Audit all components in the EDA execution path (initializer, strategy, operator functions, distribution sampling) for calls to `np.random.*` and replace them with calls to the injected `self.random_state`.
