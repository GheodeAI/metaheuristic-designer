# ARCH_SUMMARY.md — Architecture Summary

**Package:** `metaheuristic-designer` v0.4.0  
**Full critique:** see [`ARCHITECTURE.md`](ARCHITECTURE.md)  
**Defects:** see [`ERRORES.md`](ERRORES.md)

---

## What the package does

A composable Python framework for improvement-based metaheuristic optimization. Every component of the optimization loop (initialization, operators, selection, encoding, constraint handling, stopping, reporting) is an independently swappable abstraction. Supports GA, DE, ES, CMA-ES, PSO, SA, EDA, and Bayesian Optimization.

---

## Core design pattern

```
Algorithm
  └── SearchStrategy (template method: init → perturb → repair → select)
        ├── Initializer       → Population
        ├── Operator          → offspring Population
        ├── ConstraintHandler → repaired genotype
        └── SurvivorSelection → next-generation Population
```

`ParametrizableMixin` makes any numeric parameter transparently schedulable. `HistoryTracker` records per-iteration statistics. `StoppingCondition` supports compound boolean expressions over multiple criteria.

---

## Strengths

- **Composability:** plugin/lambda pattern applied consistently at every extension point
- **Algorithm breadth:** all major population-based paradigms under one interface
- **Parameter scheduling:** first-class adaptive parameters via `SchedulableParameter`
- **Reproducibility:** consistent `random_state` injection throughout (with known exceptions)
- **`simple` module:** low-barrier entry without hiding composable internals

---

## Critical defects (from `ERRORES.md`)

| # | Severity | Summary |
|---|---|---|
| 11 | **Critical** | CMA-ES always raises `AttributeError` on `initialize()` — non-functional |
| 1 | **Critical** | CMA-ES uses global `np.random` instead of injected `random_state` |
| 2 | **Critical** | MULTIGAUSS multi-individual path passes wrong kwarg to `multivariate_normal` |
| 12 | **Critical** | `multivariate_categorical.rvs()` shape mismatch breaks MULTICATEGORICAL sampling |
| 5 | **High** | `SearchStrategy` not `ABC` — incomplete subclasses accepted silently |
| 10 | **High** | `CompositeOperator` bypasses sub-operator encodings |
| 13 | **High** | `CompositeEncoding` crashes on non-`ParameterExtendingEncoding` members |
| 4 | **High** | Operator functions mutate input array in place without documented contract |

---

## Test suite status

| Metric | Value |
|---|---|
| Total tests | 489 passing, 5 xfailed (known bugs) |
| Coverage | **80.75%** (threshold: 80%) |
| Test types | Unit, property-based (Hypothesis), integration |
| Environment | `conda run -n mhd pytest` |

Run with: `conda run -n mhd pytest --cov=metaheuristic_designer --cov-report=term-missing`
