# tests/test_simple.py
import warnings
from sklearn.exceptions import ConvergenceWarning
import pytest
from conftest import dummy_objfunc, discrete_objfunc, permutation_objfunc, sphere_objfunc, run_and_get_best
import metaheuristic_designer.simple as simple

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# -------------------------------------------------------------------
# Hill Climbing
# -------------------------------------------------------------------
def test_hill_climb_binary_reproducible(dummy_objfunc):
    b1 = run_and_get_best(simple.hill_climb_binary, dummy_objfunc, seed=42)
    b2 = run_and_get_best(simple.hill_climb_binary, dummy_objfunc, seed=42)
    assert b1 == b2


def test_hill_climb_discrete_reproducible(discrete_objfunc):
    b1 = run_and_get_best(simple.hill_climb_discrete, discrete_objfunc, seed=42)
    b2 = run_and_get_best(simple.hill_climb_discrete, discrete_objfunc, seed=42)
    assert b1 == b2


def test_hill_climb_permutation_reproducible(permutation_objfunc):
    b1 = run_and_get_best(simple.hill_climb_permutation, permutation_objfunc, seed=42)
    b2 = run_and_get_best(simple.hill_climb_permutation, permutation_objfunc, seed=42)
    assert b1 == b2


def test_hill_climb_real_reproducible(sphere_objfunc):
    b1 = run_and_get_best(simple.hill_climb_real, sphere_objfunc, seed=42)
    b2 = run_and_get_best(simple.hill_climb_real, sphere_objfunc, seed=42)
    assert b1 == b2


def test_hill_climb_real_different_seeds(sphere_objfunc):
    b1 = run_and_get_best(simple.hill_climb_real, sphere_objfunc, seed=42)
    b2 = run_and_get_best(simple.hill_climb_real, sphere_objfunc, seed=123)
    assert b1 != b2


# -------------------------------------------------------------------
# Local Search
# -------------------------------------------------------------------
def test_local_search_binary_reproducible(dummy_objfunc):
    b1 = run_and_get_best(simple.local_search_binary, dummy_objfunc, seed=42)
    b2 = run_and_get_best(simple.local_search_binary, dummy_objfunc, seed=42)
    assert b1 == b2


def test_local_search_discrete_reproducible(discrete_objfunc):
    b1 = run_and_get_best(simple.local_search_discrete, discrete_objfunc, seed=42)
    b2 = run_and_get_best(simple.local_search_discrete, discrete_objfunc, seed=42)
    assert b1 == b2


def test_local_search_permutation_reproducible(permutation_objfunc):
    b1 = run_and_get_best(simple.local_search_permutation, permutation_objfunc, seed=42)
    b2 = run_and_get_best(simple.local_search_permutation, permutation_objfunc, seed=42)
    assert b1 == b2


def test_local_search_real_reproducible(sphere_objfunc):
    b1 = run_and_get_best(simple.local_search_real, sphere_objfunc, seed=42)
    b2 = run_and_get_best(simple.local_search_real, sphere_objfunc, seed=42)
    assert b1 == b2


def test_local_search_real_different_seeds(sphere_objfunc):
    b1 = run_and_get_best(simple.local_search_real, sphere_objfunc, seed=42)
    b2 = run_and_get_best(simple.local_search_real, sphere_objfunc, seed=123)
    assert b1 != b2


# -------------------------------------------------------------------
# Simulated Annealing
# -------------------------------------------------------------------
def test_simulated_annealing_binary_reproducible(dummy_objfunc):
    b1 = run_and_get_best(simple.simulated_annealing_binary, dummy_objfunc, seed=42)
    b2 = run_and_get_best(simple.simulated_annealing_binary, dummy_objfunc, seed=42)
    assert b1 == b2


def test_simulated_annealing_discrete_reproducible(discrete_objfunc):
    b1 = run_and_get_best(simple.simulated_annealing_discrete, discrete_objfunc, seed=42)
    b2 = run_and_get_best(simple.simulated_annealing_discrete, discrete_objfunc, seed=42)
    assert b1 == b2


def test_simulated_annealing_permutation_reproducible(permutation_objfunc):
    b1 = run_and_get_best(simple.simulated_annealing_permutation, permutation_objfunc, seed=42)
    b2 = run_and_get_best(simple.simulated_annealing_permutation, permutation_objfunc, seed=42)
    assert b1 == b2


def test_simulated_annealing_real_reproducible(sphere_objfunc):
    b1 = run_and_get_best(simple.simulated_annealing_real, sphere_objfunc, seed=42)
    b2 = run_and_get_best(simple.simulated_annealing_real, sphere_objfunc, seed=42)
    assert b1 == b2


def test_simulated_annealing_real_different_seeds(sphere_objfunc):
    b1 = run_and_get_best(simple.simulated_annealing_real, sphere_objfunc, seed=42)
    b2 = run_and_get_best(simple.simulated_annealing_real, sphere_objfunc, seed=123)
    assert b1 != b2


# -------------------------------------------------------------------
# Evolution Strategy
# -------------------------------------------------------------------
def test_evolution_strategy_binary_reproducible(dummy_objfunc):
    b1 = run_and_get_best(simple.evolution_strategy_binary, dummy_objfunc, seed=42)
    b2 = run_and_get_best(simple.evolution_strategy_binary, dummy_objfunc, seed=42)
    assert b1 == b2


def test_evolution_strategy_discrete_reproducible(discrete_objfunc):
    b1 = run_and_get_best(simple.evolution_strategy_discrete, discrete_objfunc, seed=42)
    b2 = run_and_get_best(simple.evolution_strategy_discrete, discrete_objfunc, seed=42)
    assert b1 == b2


def test_evolution_strategy_permutation_reproducible(permutation_objfunc):
    b1 = run_and_get_best(simple.evolution_strategy_permutation, permutation_objfunc, seed=42)
    b2 = run_and_get_best(simple.evolution_strategy_permutation, permutation_objfunc, seed=42)
    assert b1 == b2


def test_evolution_strategy_real_reproducible(sphere_objfunc):
    b1 = run_and_get_best(simple.evolution_strategy_real, sphere_objfunc, seed=42)
    b2 = run_and_get_best(simple.evolution_strategy_real, sphere_objfunc, seed=42)
    assert b1 == b2


def test_evolution_strategy_real_different_seeds(sphere_objfunc):
    b1 = run_and_get_best(simple.evolution_strategy_real, sphere_objfunc, seed=42)
    b2 = run_and_get_best(simple.evolution_strategy_real, sphere_objfunc, seed=123)
    assert b1 != b2


# -------------------------------------------------------------------
# Genetic Algorithm
# -------------------------------------------------------------------
def test_genetic_algorithm_binary_reproducible(dummy_objfunc):
    b1 = run_and_get_best(simple.genetic_algorithm_binary, dummy_objfunc, seed=42)
    b2 = run_and_get_best(simple.genetic_algorithm_binary, dummy_objfunc, seed=42)
    assert b1 == b2


def test_genetic_algorithm_discrete_reproducible(discrete_objfunc):
    b1 = run_and_get_best(simple.genetic_algorithm_discrete, discrete_objfunc, seed=42)
    b2 = run_and_get_best(simple.genetic_algorithm_discrete, discrete_objfunc, seed=42)
    assert b1 == b2


def test_genetic_algorithm_permutation_reproducible(permutation_objfunc):
    b1 = run_and_get_best(simple.genetic_algorithm_permutation, permutation_objfunc, seed=42)
    b2 = run_and_get_best(simple.genetic_algorithm_permutation, permutation_objfunc, seed=42)
    assert b1 == b2


def test_genetic_algorithm_real_reproducible(sphere_objfunc):
    b1 = run_and_get_best(simple.genetic_algorithm_real, sphere_objfunc, seed=42)
    b2 = run_and_get_best(simple.genetic_algorithm_real, sphere_objfunc, seed=42)
    assert b1 == b2


def test_genetic_algorithm_real_different_seeds(sphere_objfunc):
    b1 = run_and_get_best(simple.genetic_algorithm_real, sphere_objfunc, seed=42)
    b2 = run_and_get_best(simple.genetic_algorithm_real, sphere_objfunc, seed=123)
    assert b1 != b2


# -------------------------------------------------------------------
# Differential Evolution
# -------------------------------------------------------------------
def test_differential_evolution_bin_reproducible(dummy_objfunc):
    b1 = run_and_get_best(simple.differential_evolution_binary, dummy_objfunc, seed=42)
    b2 = run_and_get_best(simple.differential_evolution_binary, dummy_objfunc, seed=42)
    assert b1 == b2


def test_differential_evolution_int_reproducible(discrete_objfunc):
    b1 = run_and_get_best(simple.differential_evolution_discrete, discrete_objfunc, seed=42)
    b2 = run_and_get_best(simple.differential_evolution_discrete, discrete_objfunc, seed=42)
    assert b1 == b2


def test_differential_evolution_real_reproducible(sphere_objfunc):
    b1 = run_and_get_best(simple.differential_evolution_real, sphere_objfunc, seed=42)
    b2 = run_and_get_best(simple.differential_evolution_real, sphere_objfunc, seed=42)
    assert b1 == b2


def test_differential_evolution_real_different_seeds(sphere_objfunc):
    b1 = run_and_get_best(simple.differential_evolution_real, sphere_objfunc, seed=42)
    b2 = run_and_get_best(simple.differential_evolution_real, sphere_objfunc, seed=123)
    assert b1 != b2


# -------------------------------------------------------------------
# Particle Swarm
# -------------------------------------------------------------------
def test_particle_swarm_bin_reproducible(sphere_objfunc):
    b1 = run_and_get_best(simple.particle_swarm_binary, sphere_objfunc, seed=42)
    b2 = run_and_get_best(simple.particle_swarm_binary, sphere_objfunc, seed=42)
    assert b1 == b2


def test_particle_swarm_int_reproducible(discrete_objfunc):
    b1 = run_and_get_best(simple.particle_swarm_discrete, discrete_objfunc, seed=42)
    b2 = run_and_get_best(simple.particle_swarm_discrete, discrete_objfunc, seed=42)
    assert b1 == b2


def test_particle_swarm_real_reproducible(sphere_objfunc):
    b1 = run_and_get_best(simple.particle_swarm_real, sphere_objfunc, seed=42)
    b2 = run_and_get_best(simple.particle_swarm_real, sphere_objfunc, seed=42)
    assert b1 == b2


def test_particle_swarm_real_different_seeds(sphere_objfunc):
    b1 = run_and_get_best(simple.particle_swarm_real, sphere_objfunc, seed=42)
    b2 = run_and_get_best(simple.particle_swarm_real, sphere_objfunc, seed=123)
    assert b1 != b2


# -------------------------------------------------------------------
# Bayesian Optimization (only real)
# -------------------------------------------------------------------
def test_bayesian_optimization_real_reproducible(sphere_objfunc):
    b1 = run_and_get_best(simple.bayesian_optimization_real, sphere_objfunc, seed=42)
    b2 = run_and_get_best(simple.bayesian_optimization_real, sphere_objfunc, seed=42)
    assert b1 == b2


def test_bayesian_optimization_real_different_seeds(sphere_objfunc):
    b1 = run_and_get_best(simple.bayesian_optimization_real, sphere_objfunc, seed=42)
    b2 = run_and_get_best(simple.bayesian_optimization_real, sphere_objfunc, seed=123)
    assert b1 != b2


# -------------------------------------------------------------------
# Random Search
# -------------------------------------------------------------------
def test_random_search_binary_reproducible(dummy_objfunc):
    b1 = run_and_get_best(simple.random_search_binary, dummy_objfunc, seed=42)
    b2 = run_and_get_best(simple.random_search_binary, dummy_objfunc, seed=42)
    assert b1 == b2


def test_random_search_discrete_reproducible(discrete_objfunc):
    b1 = run_and_get_best(simple.random_search_discrete, discrete_objfunc, seed=42)
    b2 = run_and_get_best(simple.random_search_discrete, discrete_objfunc, seed=42)
    assert b1 == b2


def test_random_search_permutation_reproducible(permutation_objfunc):
    b1 = run_and_get_best(simple.random_search_permutation, permutation_objfunc, seed=42)
    b2 = run_and_get_best(simple.random_search_permutation, permutation_objfunc, seed=42)
    assert b1 == b2


def test_random_search_real_reproducible(sphere_objfunc):
    b1 = run_and_get_best(simple.random_search_real, sphere_objfunc, seed=42)
    b2 = run_and_get_best(simple.random_search_real, sphere_objfunc, seed=42)
    assert b1 == b2


def test_random_search_real_different_seeds(sphere_objfunc):
    b1 = run_and_get_best(simple.random_search_real, sphere_objfunc, seed=42)
    b2 = run_and_get_best(simple.random_search_real, sphere_objfunc, seed=123)
    assert b1 != b2
