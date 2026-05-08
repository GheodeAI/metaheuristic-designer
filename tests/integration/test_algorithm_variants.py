"""
Integration tests for less-exercised algorithm variants in the simple module.

Covers: binary/permutation/discrete ES; permutation/discrete HC, SA, RS, LS;
        permutation GA; discrete/permutation DE; binary DE; binary PSO;
        permutation ES; discrete ES.
"""

import numpy as np
import pytest

from metaheuristic_designer.benchmarks import MaxOnes, Sphere
from metaheuristic_designer import simple

COMMON = {
    "stop_cond": "max_iterations",
    "max_iterations": 50,
    "reporter": "silent",
}


def run(algo):
    pop = algo.optimize()
    sol, fit = pop.best_solution()
    return sol, fit


# ---------------------------------------------------------------------------
# Simple benchmark with bounded integer domain for discrete algorithms
# ---------------------------------------------------------------------------

class _BoundedIntObj:
    """Minimal ObjectiveFunc-like object for discrete hill climbing tests."""
    pass


def _make_discrete_sphere():
    """Sphere(min) works as a discrete obj if we cast to int."""
    return Sphere(dimension=4, mode="min")


# ---------------------------------------------------------------------------
# Evolution Strategy variants
# ---------------------------------------------------------------------------

def test_es_binary_maxones_runs():
    objfunc = MaxOnes(dimension=8)
    algo = simple.evolution_strategy_binary(objfunc, population_size=20, offspring_size=40,
                                            random_state=0, **COMMON)
    sol, fit = run(algo)
    assert sol is not None and np.isfinite(fit)


def test_es_binary_elitist_maxones_runs():
    objfunc = MaxOnes(dimension=8)
    algo = simple.evolution_strategy_binary(objfunc, population_size=20, offspring_size=40,
                                            elitist=True, random_state=1, **COMMON)
    sol, fit = run(algo)
    assert sol is not None and np.isfinite(fit)


# ---------------------------------------------------------------------------
# Hill Climbing variants
# ---------------------------------------------------------------------------

def test_hill_climb_permutation_runs():
    objfunc = MaxOnes(dimension=6)
    algo = simple.hill_climb_permutation(objfunc, random_state=0, **COMMON)
    sol, fit = run(algo)
    assert sol is not None and np.isfinite(fit)


def test_hill_climb_discrete_runs():
    objfunc = Sphere(dimension=4, mode="min")
    algo = simple.hill_climb_discrete(objfunc, random_state=0, **COMMON)
    sol, fit = run(algo)
    assert sol is not None and np.isfinite(fit)


# ---------------------------------------------------------------------------
# Simulated Annealing variants
# ---------------------------------------------------------------------------

def test_sa_permutation_runs():
    objfunc = MaxOnes(dimension=6)
    algo = simple.simulated_annealing_permutation(objfunc, random_state=0, **COMMON)
    sol, fit = run(algo)
    assert sol is not None and np.isfinite(fit)


def test_sa_discrete_runs():
    objfunc = Sphere(dimension=4, mode="min")
    algo = simple.simulated_annealing_discrete(objfunc, random_state=0, **COMMON)
    sol, fit = run(algo)
    assert sol is not None and np.isfinite(fit)


# ---------------------------------------------------------------------------
# Random Search variants
# ---------------------------------------------------------------------------

def test_random_search_permutation_runs():
    objfunc = MaxOnes(dimension=6)
    algo = simple.random_search_permutation(objfunc, random_state=0, **COMMON)
    sol, fit = run(algo)
    assert sol is not None and np.isfinite(fit)


def test_random_search_discrete_runs():
    objfunc = Sphere(dimension=4, mode="min")
    algo = simple.random_search_discrete(objfunc, random_state=0, **COMMON)
    sol, fit = run(algo)
    assert sol is not None and np.isfinite(fit)


# ---------------------------------------------------------------------------
# Local Search variants
# ---------------------------------------------------------------------------

def test_local_search_permutation_runs():
    objfunc = MaxOnes(dimension=6)
    algo = simple.local_search_permutation(objfunc, random_state=0, **COMMON)
    sol, fit = run(algo)
    assert sol is not None and np.isfinite(fit)


def test_local_search_discrete_runs():
    objfunc = Sphere(dimension=4, mode="min")
    algo = simple.local_search_discrete(objfunc, random_state=0, **COMMON)
    sol, fit = run(algo)
    assert sol is not None and np.isfinite(fit)


# ---------------------------------------------------------------------------
# Genetic Algorithm variants
# ---------------------------------------------------------------------------

def test_ga_permutation_runs():
    objfunc = MaxOnes(dimension=6)
    algo = simple.genetic_algorithm_permutation(objfunc, population_size=20,
                                                random_state=0, **COMMON)
    sol, fit = run(algo)
    assert sol is not None and np.isfinite(fit)


def test_ga_discrete_runs():
    objfunc = Sphere(dimension=4, mode="min")
    algo = simple.genetic_algorithm_discrete(objfunc, population_size=20,
                                             random_state=0, **COMMON)
    sol, fit = run(algo)
    assert sol is not None and np.isfinite(fit)


# ---------------------------------------------------------------------------
# Particle Swarm variants
# ---------------------------------------------------------------------------

def test_pso_discrete_runs():
    objfunc = Sphere(dimension=4, mode="min")
    algo = simple.particle_swarm_discrete(objfunc, random_state=0, **COMMON)
    sol, fit = run(algo)
    assert sol is not None and np.isfinite(fit)


# ---------------------------------------------------------------------------
# Differential Evolution variants (discrete)
# ---------------------------------------------------------------------------

def test_de_discrete_runs():
    objfunc = Sphere(dimension=4, mode="min")
    algo = simple.differential_evolution_discrete(objfunc, random_state=0, **COMMON)
    sol, fit = run(algo)
    assert sol is not None and np.isfinite(fit)


def test_de_binary_maxones_runs():
    objfunc = MaxOnes(dimension=8)
    algo = simple.differential_evolution_binary(objfunc, population_size=20, random_state=0, **COMMON)
    sol, fit = run(algo)
    assert sol is not None and np.isfinite(fit)


# ---------------------------------------------------------------------------
# Particle Swarm binary variant
# ---------------------------------------------------------------------------

def test_pso_binary_maxones_runs():
    objfunc = MaxOnes(dimension=8)
    algo = simple.particle_swarm_binary(objfunc, population_size=20, random_state=0, **COMMON)
    sol, fit = run(algo)
    assert sol is not None and np.isfinite(fit)


# ---------------------------------------------------------------------------
# Evolution Strategy permutation and discrete variants
# ---------------------------------------------------------------------------

def test_es_permutation_maxones_runs():
    objfunc = MaxOnes(dimension=6)
    algo = simple.evolution_strategy_permutation(objfunc, population_size=20, offspring_size=40,
                                                 random_state=0, **COMMON)
    sol, fit = run(algo)
    assert sol is not None and np.isfinite(fit)


def test_es_permutation_elitist_runs():
    objfunc = MaxOnes(dimension=6)
    algo = simple.evolution_strategy_permutation(objfunc, population_size=20, offspring_size=40,
                                                 elitist=True, random_state=1, **COMMON)
    sol, fit = run(algo)
    assert sol is not None and np.isfinite(fit)


def test_es_discrete_sphere_runs():
    objfunc = Sphere(dimension=4, mode="min")
    algo = simple.evolution_strategy_discrete(objfunc, population_size=20, offspring_size=40,
                                              random_state=0, **COMMON)
    sol, fit = run(algo)
    assert sol is not None and np.isfinite(fit)


def test_es_discrete_elitist_runs():
    objfunc = Sphere(dimension=4, mode="min")
    algo = simple.evolution_strategy_discrete(objfunc, population_size=20, offspring_size=40,
                                              elitist=True, random_state=2, **COMMON)
    sol, fit = run(algo)
    assert sol is not None and np.isfinite(fit)
