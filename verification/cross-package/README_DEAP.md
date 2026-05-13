# DEAP Wrapper for metaheuristic-designer

This folder contains a duck‑typed wrapper (`DEAPWrapper`) that
makes any DEAP algorithm usable inside the `run_experiment` benchmark
loop.

## What the wrapper provides

* `optimize()` – runs the DEAP algorithm (default `eaSimple`) with
  the given toolbox, statistics, and hall‑of‑fame.
* `best_solution()` – returns the best phenotype and its raw
  objective value.
* `history_tracker.to_pandas()` – a DataFrame with generation‑wise
  best‑so‑far values (per‑generation convergence).

## How to use it (minimal example)

```python
from metaheuristic_designer.wrappers.deap_wrapper import DEAPWrapper

# 1. Define a builder that returns (toolbox, pop, stats, hof)
def my_builder(objfunc, seed):
    # … create DEAP toolbox, pop, stats, hof …
    return toolbox, pop, stats, hof

# 2. Wrap it
solver = DEAPWrapper(objfunc, build_fn=my_builder, ngen=100, seed=42)
solver.optimize()
best_x, best_obj = solver.best_solution()
```

## Canonical algorithm factories (used in the paper)

The file `experiments/deap_canonical.py` (gitignored until publication)
provides factory functions for three DEAP algorithms:

| Function | Algorithm | Canonical Parameters |
| :--- | :--- | :--- |
| `canonical_deap_ga()` | Genetic Algorithm | `cxpb=0.7`, `mutpb=0.3`, tournament size 3, pop 100 |
| `canonical_deap_es()` | (μ+λ) Evolution Strategy | `cxpb=0.0`, `mutpb=1.0`, pop 500 |
| `canonical_deap_de()` | Differential Evolution (DE/rand/1) | `F=0.8`, `CR=0.9`, pop 50 |

For Particle Swarm Optimisation, we recommend using the **PyGMO** or
**Nevergrad** wrappers, which provide native, well‑tested PSO
implementations.

## Dependency

`pip install deap`