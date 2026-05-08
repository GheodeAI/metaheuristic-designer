"""
Integration tests for Simulated Annealing.

Contract:
- Algorithm must complete without error.
- Solution must have the correct dimension.
- Fitness must be a finite number.
- Results must be reproducible with a fixed seed.
- For maximisation, fitness should be at least as good as a random solution
  after a reasonable number of iterations.
"""

import numpy as np
import pytest

from metaheuristic_designer.benchmarks import Sphere, MaxOnes
from metaheuristic_designer import simple


COMMON = {
    "stop_cond": "max_iterations",
    "max_iterations": 300,
    "reporter": "silent",
}


def run_and_get_fitness(algo):
    population = algo.optimize()
    _solution, fitness = population.best_solution()
    return fitness


# ---------------------------------------------------------------------------
# SA on real-valued Sphere minimisation
# ---------------------------------------------------------------------------

def test_sa_real_sphere_runs_without_error():
    objfunc = Sphere(dimension=3, mode="min")
    algo = simple.simulated_annealing_real(objfunc, random_state=0, **COMMON)
    population = algo.optimize()
    solution, fitness = population.best_solution()
    assert solution is not None
    assert np.isfinite(fitness)


def test_sa_real_sphere_solution_dimension():
    objfunc = Sphere(dimension=4, mode="min")
    algo = simple.simulated_annealing_real(objfunc, random_state=1, **COMMON)
    population = algo.optimize()
    solution, _ = population.best_solution()
    assert solution.shape == (4,)


def test_sa_real_sphere_reproducible():
    objfunc = Sphere(dimension=3, mode="min")
    f1 = run_and_get_fitness(simple.simulated_annealing_real(objfunc, random_state=99, **COMMON))
    f2 = run_and_get_fitness(simple.simulated_annealing_real(objfunc, random_state=99, **COMMON))
    assert f1 == pytest.approx(f2)


def test_sa_real_sphere_fitness_is_finite():
    objfunc = Sphere(dimension=3, mode="min")
    algo = simple.simulated_annealing_real(objfunc, random_state=2, **COMMON)
    fitness = run_and_get_fitness(algo)
    assert np.isfinite(fitness)


# ---------------------------------------------------------------------------
# SA on binary MaxOnes
# ---------------------------------------------------------------------------

def test_sa_binary_maxones_runs():
    objfunc = MaxOnes(dimension=8)
    algo = simple.simulated_annealing_binary(objfunc, random_state=3, **COMMON)
    population = algo.optimize()
    solution, fitness = population.best_solution()
    assert solution is not None
    assert np.isfinite(fitness)


def test_sa_binary_maxones_reproducible():
    objfunc = MaxOnes(dimension=8)
    f1 = run_and_get_fitness(simple.simulated_annealing_binary(objfunc, random_state=11, **COMMON))
    f2 = run_and_get_fitness(simple.simulated_annealing_binary(objfunc, random_state=11, **COMMON))
    assert f1 == pytest.approx(f2)
