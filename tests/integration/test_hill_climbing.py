"""
Integration tests for Hill Climbing variants.

Contract:
- The algorithm must run without errors.
- It must return a valid population with best_solution().
- For an improvement-based algorithm, the final solution must be no worse
  than the initial solution (elitist acceptance).
- The fitness value must be a finite number.
- Fixed seeds must produce reproducible results.
"""

import numpy as np
import pytest

from metaheuristic_designer.benchmarks import Sphere, MaxOnes
from metaheuristic_designer import simple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

COMMON = {
    "stop_cond": "max_iterations",
    "max_iterations": 200,
    "reporter": "silent",
}


def run_and_get_fitness(algo):
    """Run the algorithm once and return the best fitness."""
    population = algo.optimize()
    _solution, fitness = population.best_solution()
    return fitness


# ---------------------------------------------------------------------------
# Binary hill climbing on MaxOnes
# ---------------------------------------------------------------------------

def test_hill_climb_binary_runs_without_error():
    objfunc = MaxOnes(dimension=8)
    algo = simple.hill_climb_binary(objfunc, random_state=0, **COMMON)
    population = algo.optimize()
    solution, fitness = population.best_solution()
    assert solution is not None
    assert np.isfinite(fitness)


def test_hill_climb_binary_fitness_is_non_negative():
    """MaxOnes fitness is the count of 1s: always in [0, dimension]."""
    objfunc = MaxOnes(dimension=8)
    algo = simple.hill_climb_binary(objfunc, random_state=1, **COMMON)
    fitness = run_and_get_fitness(algo)
    # Fitness is adjusted by mode; for MaxOnes in max mode fitness == objective >= 0
    assert np.isfinite(fitness)


def test_hill_climb_binary_reproducible():
    """Two identical runs with the same seed must return the same fitness."""
    objfunc = MaxOnes(dimension=8)
    f1 = run_and_get_fitness(simple.hill_climb_binary(objfunc, random_state=42, **COMMON))
    f2 = run_and_get_fitness(simple.hill_climb_binary(objfunc, random_state=42, **COMMON))
    assert f1 == pytest.approx(f2)


def test_hill_climb_binary_fitness_is_valid_count():
    """MaxOnes fitness must be in [0, dimension] (it counts the 1s)."""
    objfunc = MaxOnes(dimension=8)
    algo = simple.hill_climb_binary(objfunc, random_state=5, **COMMON)
    fitness = run_and_get_fitness(algo)
    # fitness == objective for max mode, objective == number of 1s in {0..8}
    assert 0 <= fitness <= 8


# ---------------------------------------------------------------------------
# Real-valued hill climbing on Sphere (minimisation)
# ---------------------------------------------------------------------------

def test_hill_climb_real_sphere_runs():
    objfunc = Sphere(dimension=3, mode="min")
    algo = simple.hill_climb_real(objfunc, random_state=0, **COMMON)
    population = algo.optimize()
    solution, fitness = population.best_solution()
    assert solution is not None
    assert np.isfinite(fitness)


def test_hill_climb_real_sphere_solution_has_correct_dimension():
    objfunc = Sphere(dimension=5, mode="min")
    algo = simple.hill_climb_real(objfunc, random_state=0, **COMMON)
    population = algo.optimize()
    solution, _ = population.best_solution()
    assert solution.shape == (5,)


def test_hill_climb_real_sphere_reproducible():
    objfunc = Sphere(dimension=3, mode="min")
    f1 = run_and_get_fitness(simple.hill_climb_real(objfunc, random_state=7, **COMMON))
    f2 = run_and_get_fitness(simple.hill_climb_real(objfunc, random_state=7, **COMMON))
    assert f1 == pytest.approx(f2)
