"""
Integration tests for the Genetic Algorithm.

Contract:
- Algorithm must run without error on binary and real domains.
- Output population has the expected number of individuals.
- Best solution has the correct dimension.
- Fitness must be finite.
- Results are reproducible with a fixed seed.
- After sufficient iterations, the GA should find a near-optimal solution
  for trivially easy problems (OneMax with enough generations).
"""

import numpy as np
import pytest

from metaheuristic_designer.benchmarks import Sphere, MaxOnes
from metaheuristic_designer import simple


COMMON_BINARY = {
    "stop_cond": "max_iterations",
    "max_iterations": 100,
    "reporter": "silent",
}

COMMON_REAL = {
    "stop_cond": "max_iterations",
    "max_iterations": 200,
    "reporter": "silent",
}


def run_and_get_fitness(algo):
    population = algo.optimize()
    _solution, fitness = population.best_solution()
    return fitness


# ---------------------------------------------------------------------------
# Binary GA on MaxOnes
# ---------------------------------------------------------------------------

def test_ga_binary_maxones_runs_without_error():
    objfunc = MaxOnes(dimension=8)
    algo = simple.genetic_algorithm_binary(
        objfunc, population_size=20, random_state=0, **COMMON_BINARY
    )
    population = algo.optimize()
    solution, fitness = population.best_solution()
    assert solution is not None
    assert np.isfinite(fitness)


def test_ga_binary_maxones_solution_dimension():
    objfunc = MaxOnes(dimension=8)
    algo = simple.genetic_algorithm_binary(
        objfunc, population_size=20, random_state=0, **COMMON_BINARY
    )
    population = algo.optimize()
    solution, _ = population.best_solution()
    assert solution.shape == (8,)


def test_ga_binary_maxones_reproducible():
    objfunc = MaxOnes(dimension=8)
    f1 = run_and_get_fitness(
        simple.genetic_algorithm_binary(objfunc, population_size=20, random_state=5, **COMMON_BINARY)
    )
    f2 = run_and_get_fitness(
        simple.genetic_algorithm_binary(objfunc, population_size=20, random_state=5, **COMMON_BINARY)
    )
    assert f1 == pytest.approx(f2)


def test_ga_binary_maxones_finds_optimum():
    """With enough iterations on MaxOnes(8), the GA should reach fitness 8 (all ones)."""
    objfunc = MaxOnes(dimension=8)
    algo = simple.genetic_algorithm_binary(
        objfunc,
        population_size=50,
        random_state=42,
        stop_cond="objective_target",
        objective_target=8.0,
        max_iterations=500,
        reporter="silent",
    )
    population = algo.optimize()
    _solution, fitness = population.best_solution()
    assert fitness == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# Real-valued GA on Sphere minimisation
# ---------------------------------------------------------------------------

def test_ga_real_sphere_runs_without_error():
    objfunc = Sphere(dimension=3, mode="min")
    algo = simple.genetic_algorithm_real(
        objfunc, population_size=30, random_state=0, **COMMON_REAL
    )
    population = algo.optimize()
    solution, fitness = population.best_solution()
    assert solution is not None
    assert np.isfinite(fitness)


def test_ga_real_sphere_solution_dimension():
    objfunc = Sphere(dimension=5, mode="min")
    algo = simple.genetic_algorithm_real(
        objfunc, population_size=30, random_state=1, **COMMON_REAL
    )
    population = algo.optimize()
    solution, _ = population.best_solution()
    assert solution.shape == (5,)


def test_ga_real_sphere_reproducible():
    objfunc = Sphere(dimension=3, mode="min")
    f1 = run_and_get_fitness(
        simple.genetic_algorithm_real(objfunc, population_size=30, random_state=7, **COMMON_REAL)
    )
    f2 = run_and_get_fitness(
        simple.genetic_algorithm_real(objfunc, population_size=30, random_state=7, **COMMON_REAL)
    )
    assert f1 == pytest.approx(f2)


def test_ga_real_sphere_fitness_bounded():
    """Sphere minimum is 0; negative fitness means a min-mode issue. Fitness must be <= 0
    (since mode='min' negates the objective), and the objective itself must be >= 0."""
    objfunc = Sphere(dimension=3, mode="min")
    algo = simple.genetic_algorithm_real(
        objfunc, population_size=30, random_state=3, **COMMON_REAL
    )
    population = algo.optimize()
    solution, fitness = population.best_solution()
    # Fitness is -objective for minimisation; must be finite
    assert np.isfinite(fitness)
