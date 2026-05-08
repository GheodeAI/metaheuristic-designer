"""
Integration tests for ES, PSO, local search, and random search.

Contract:
- All algorithms must run without error.
- Best solution must have correct dimension.
- Fitness must be finite.
- Reproducible with fixed seed.
"""

import numpy as np
import pytest

from metaheuristic_designer.benchmarks import Sphere, MaxOnes
from metaheuristic_designer import simple


COMMON = {
    "stop_cond": "max_iterations",
    "max_iterations": 100,
    "reporter": "silent",
}


def run_and_get(algo):
    population = algo.optimize()
    solution, fitness = population.best_solution()
    return solution, fitness


# ---------------------------------------------------------------------------
# Evolution Strategy (ES)
# ---------------------------------------------------------------------------

def test_es_real_sphere_runs():
    objfunc = Sphere(dimension=4, mode="min")
    algo = simple.evolution_strategy_real(objfunc, random_state=0, **COMMON)
    solution, fitness = run_and_get(algo)
    assert solution is not None
    assert np.isfinite(fitness)


def test_es_real_sphere_solution_dimension():
    objfunc = Sphere(dimension=5, mode="min")
    algo = simple.evolution_strategy_real(objfunc, random_state=1, **COMMON)
    population = algo.optimize()
    solution, _ = population.best_solution()
    assert solution.shape == (5,)


def test_es_real_sphere_reproducible():
    objfunc = Sphere(dimension=4, mode="min")
    f1 = run_and_get(simple.evolution_strategy_real(objfunc, random_state=9, **COMMON))[1]
    f2 = run_and_get(simple.evolution_strategy_real(objfunc, random_state=9, **COMMON))[1]
    assert f1 == pytest.approx(f2)


# ---------------------------------------------------------------------------
# Particle Swarm Optimization (PSO)
# ---------------------------------------------------------------------------

def test_pso_real_sphere_runs():
    objfunc = Sphere(dimension=4, mode="min")
    algo = simple.particle_swarm_real(objfunc, random_state=0, **COMMON)
    solution, fitness = run_and_get(algo)
    assert solution is not None
    assert np.isfinite(fitness)


def test_pso_real_sphere_solution_dimension():
    objfunc = Sphere(dimension=5, mode="min")
    algo = simple.particle_swarm_real(objfunc, random_state=2, **COMMON)
    population = algo.optimize()
    solution, _ = population.best_solution()
    assert solution.shape == (5,)


def test_pso_real_sphere_reproducible():
    objfunc = Sphere(dimension=3, mode="min")
    f1 = run_and_get(simple.particle_swarm_real(objfunc, random_state=7, **COMMON))[1]
    f2 = run_and_get(simple.particle_swarm_real(objfunc, random_state=7, **COMMON))[1]
    assert f1 == pytest.approx(f2)


# ---------------------------------------------------------------------------
# Random Search
# ---------------------------------------------------------------------------

def test_random_search_real_runs():
    objfunc = Sphere(dimension=4, mode="min")
    algo = simple.random_search_real(objfunc, random_state=0, **COMMON)
    solution, fitness = run_and_get(algo)
    assert solution is not None
    assert np.isfinite(fitness)


def test_random_search_binary_maxones_runs():
    objfunc = MaxOnes(dimension=8)
    algo = simple.random_search_binary(objfunc, random_state=0, **COMMON)
    solution, fitness = run_and_get(algo)
    assert solution is not None
    assert np.isfinite(fitness)


def test_random_search_real_reproducible():
    objfunc = Sphere(dimension=4, mode="min")
    f1 = run_and_get(simple.random_search_real(objfunc, random_state=3, **COMMON))[1]
    f2 = run_and_get(simple.random_search_real(objfunc, random_state=3, **COMMON))[1]
    assert f1 == pytest.approx(f2)


# ---------------------------------------------------------------------------
# Local Search
# ---------------------------------------------------------------------------

def test_local_search_real_runs():
    objfunc = Sphere(dimension=3, mode="min")
    algo = simple.local_search_real(objfunc, random_state=0, **COMMON)
    solution, fitness = run_and_get(algo)
    assert solution is not None
    assert np.isfinite(fitness)


def test_local_search_binary_maxones_runs():
    objfunc = MaxOnes(dimension=8)
    algo = simple.local_search_binary(objfunc, random_state=0, **COMMON)
    solution, fitness = run_and_get(algo)
    assert solution is not None
    assert np.isfinite(fitness)


def test_local_search_real_reproducible():
    objfunc = Sphere(dimension=3, mode="min")
    f1 = run_and_get(simple.local_search_real(objfunc, random_state=11, **COMMON))[1]
    f2 = run_and_get(simple.local_search_real(objfunc, random_state=11, **COMMON))[1]
    assert f1 == pytest.approx(f2)
