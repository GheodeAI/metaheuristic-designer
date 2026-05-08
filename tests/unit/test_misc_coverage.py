"""
Miscellaneous unit tests targeting specific uncovered lines.

Covers:
- NoSearch strategy: construction and perturb
- compute_statistic: average, median, std branches
- random_initialize and random_reset operator functions
- SeedDetermInitializer.generate_population (resets inserted counter)
- strategies.no_search.NoSearch
"""

import numpy as np
import pytest

from metaheuristic_designer.benchmarks import Sphere, MaxOnes
from metaheuristic_designer.initializers import UniformInitializer
from metaheuristic_designer.population import Population


# ---------------------------------------------------------------------------
# NoSearch strategy
# ---------------------------------------------------------------------------

def test_no_search_construction():
    from metaheuristic_designer.strategies.no_search import NoSearch
    init = UniformInitializer(dimension=4, lower_bound=-1.0, upper_bound=1.0, pop_size=5)
    strategy = NoSearch(initializer=init)
    assert strategy is not None


def test_no_search_perturb_returns_same_population():
    from metaheuristic_designer.strategies.no_search import NoSearch
    objfunc = Sphere(dimension=4, mode="min")
    init = UniformInitializer(dimension=4, lower_bound=-1.0, upper_bound=1.0, pop_size=5,
                               random_state=0)
    strategy = NoSearch(initializer=init)
    pop = init.generate_population(objfunc)
    pop.calculate_fitness()

    result = strategy.perturb(pop)
    assert result is pop


def test_no_search_algorithm_runs():
    """NoSearch can be used in a full Algorithm run."""
    from metaheuristic_designer.strategies.no_search import NoSearch
    from metaheuristic_designer.algorithm import Algorithm
    objfunc = Sphere(dimension=4, mode="min")
    init = UniformInitializer(dimension=4, lower_bound=-1.0, upper_bound=1.0, pop_size=10,
                               random_state=0)
    strategy = NoSearch(initializer=init)
    algo = Algorithm(objfunc, strategy, stop_cond="max_iterations", max_iterations=5,
                     reporter="silent")
    pop = algo.optimize()
    sol, fit = pop.best_solution()
    assert sol is not None
    assert np.isfinite(fit)


# ---------------------------------------------------------------------------
# compute_statistic: average, median, std branches
# ---------------------------------------------------------------------------

def test_compute_statistic_mean():
    from metaheuristic_designer.operators.operator_functions.random_generation import compute_statistic
    pop = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = compute_statistic(pop, stat_name="mean")
    expected = np.array([2.5, 3.5, 4.5])
    np.testing.assert_array_almost_equal(result, expected)


def test_compute_statistic_average_with_weights():
    from metaheuristic_designer.operators.operator_functions.random_generation import compute_statistic
    pop = np.array([[1.0, 2.0], [3.0, 4.0]])
    weights = np.array([1.0, 3.0])  # second row gets 3x weight
    result = compute_statistic(pop, stat_name="average", weights=weights)
    assert result is not None
    assert len(result) == 2


def test_compute_statistic_average_without_weights():
    from metaheuristic_designer.operators.operator_functions.random_generation import compute_statistic
    pop = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = compute_statistic(pop, stat_name="average")
    expected = np.array([2.0, 3.0])
    np.testing.assert_array_almost_equal(result, expected)


def test_compute_statistic_median():
    from metaheuristic_designer.operators.operator_functions.random_generation import compute_statistic
    pop = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    result = compute_statistic(pop, stat_name="median")
    expected = np.array([2.0, 5.0])
    np.testing.assert_array_almost_equal(result, expected)


def test_compute_statistic_std():
    from metaheuristic_designer.operators.operator_functions.random_generation import compute_statistic
    pop = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = compute_statistic(pop, stat_name="std")
    assert result is not None
    assert len(result) == 2


# ---------------------------------------------------------------------------
# random_initialize and random_reset
# ---------------------------------------------------------------------------

def test_random_initialize_shape_preserved():
    from metaheuristic_designer.operators.operator_functions.random_generation import random_initialize
    init = UniformInitializer(dimension=4, lower_bound=-1.0, upper_bound=1.0, pop_size=5,
                               random_state=0)
    pop = np.ones((5, 4))
    result = random_initialize(pop, initializer=init)
    assert result.shape == pop.shape


def test_random_initialize_different_from_original():
    from metaheuristic_designer.operators.operator_functions.random_generation import random_initialize
    init = UniformInitializer(dimension=4, lower_bound=-5.0, upper_bound=-4.0, pop_size=5,
                               random_state=42)
    pop = np.zeros((5, 4))  # all zeros; randomized values will be in [-5, -4]
    result = random_initialize(pop, initializer=init)
    # All results should be in [-5, -4] range
    assert np.all(result <= -4.0)
    assert np.all(result >= -5.0)


def test_random_reset_shape_preserved():
    from metaheuristic_designer.operators.operator_functions.random_generation import random_reset
    init = UniformInitializer(dimension=4, lower_bound=-1.0, upper_bound=1.0, pop_size=5,
                               random_state=0)
    pop = np.ones((5, 4))
    result = random_reset(pop, initializer=init, random_state=0, n=2)
    assert result.shape == pop.shape


# ---------------------------------------------------------------------------
# SeedDetermInitializer.generate_population (resets the counter)
# ---------------------------------------------------------------------------

def test_seed_determ_initializer_generate_population_resets():
    """generate_population resets the inserted counter so seeds start fresh each call."""
    from metaheuristic_designer.initializers.seed_initializer import SeedDetermInitializer
    base_init = UniformInitializer(dimension=4, lower_bound=-10.0, upper_bound=-9.0, pop_size=6,
                                    random_state=5)
    seed_vecs = np.array([[100.0, 200.0, 300.0, 400.0]])
    init = SeedDetermInitializer(default_init=base_init, solutions=seed_vecs, random_state=5)
    objfunc = Sphere(dimension=4, mode="min")
    pop1 = init.generate_population(objfunc, n_individuals=3)
    pop2 = init.generate_population(objfunc, n_individuals=3)
    # First individual in both populations must be the seeded solution
    np.testing.assert_array_equal(pop1.genotype_matrix[0], seed_vecs[0])
    np.testing.assert_array_equal(pop2.genotype_matrix[0], seed_vecs[0])


# ---------------------------------------------------------------------------
# Additional benchmark: ModSchwefel with extreme values (uncovered branches)
# ---------------------------------------------------------------------------

def test_mod_schwefel_extreme_high():
    """Test ModSchwefel with values > 500 (after shift)."""
    from metaheuristic_designer.benchmarks.benchmark_funcs import ModSchwefel
    obj = ModSchwefel(dimension=2, mode="min")
    # z = solution[i] + 420.97... > 500 → need solution[i] > ~79
    pop = Population(obj, np.array([[200.0, -700.0]]))
    pop.calculate_fitness()
    assert np.isfinite(pop.objective[0])


# ---------------------------------------------------------------------------
# ThreeSAT: from_cnf_file path (requires a temp file)
# ---------------------------------------------------------------------------

def test_3sat_from_cnf_file(tmp_path):
    """ThreeSAT.from_cnf_file reads a CNF file and builds a ThreeSAT problem."""
    from metaheuristic_designer.benchmarks.classic_problems import ThreeSAT

    cnf_content = "c comment\np cnf 3 2\n1 2 -3 0\n-1 2 3 0\n"
    cnf_file = tmp_path / "test.cnf"
    cnf_file.write_text(cnf_content)

    obj = ThreeSAT.from_cnf_file(cnf_file)
    assert obj.n_vars == 3
    assert obj.clauses.shape == (2, 3)


# ---------------------------------------------------------------------------
# TSP.from_csv path (requires a temp CSV file)
# ---------------------------------------------------------------------------

def test_tsp_from_csv(tmp_path):
    """TSP.from_csv reads a CSV file and builds a TSP problem."""
    from metaheuristic_designer.benchmarks.classic_problems import TSP

    csv_content = "Edge1,Edge2,Weight\n0,1,1.0\n1,2,2.0\n2,3,1.5\n3,0,2.5\n"
    csv_file = tmp_path / "tsp.csv"
    csv_file.write_text(csv_content)

    obj = TSP.from_csv(csv_file, name="TestTSP")
    assert obj is not None
    assert obj.dimension >= 4
