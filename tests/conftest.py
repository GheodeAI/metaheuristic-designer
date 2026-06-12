"""
Shared fixtures and mocks for all package tests.
No classes are defined inside test modules; everything lives here.
"""

import numpy as np
import pytest
from copy import copy
from typing import Optional, Iterable, Any, Tuple, List

# -------------------------------------------------------------------
# Imports from the package
# -------------------------------------------------------------------
from metaheuristic_designer.algorithm import Algorithm
from metaheuristic_designer.encoding import (
    Encoding,
    DefaultEncoding,
    EncodingFromLambda,
)
from metaheuristic_designer.encodings.parameter_extending_encoding import (
    ParameterExtendingEncoding,
)
from metaheuristic_designer.history_tracker import ConfigurableHistoryTracker
from metaheuristic_designer.population import Population
from metaheuristic_designer.objective_function import (
    ObjectiveFunc,
    ObjectiveFunc,
)
from metaheuristic_designer.initializer import Initializer
from metaheuristic_designer.operator import Operator, NullOperator
from metaheuristic_designer.parent_selection_base import (
    ParentSelection,
    NullParentSelection,
)
from metaheuristic_designer.survivor_selection_base import (
    SurvivorSelection,
    NullSurvivorSelection,
)
from metaheuristic_designer.utils import check_rng
from metaheuristic_designer.encodings import PSOEncoding
from metaheuristic_designer.search_strategy import SearchStrategy
from metaheuristic_designer.benchmarks.benchmark_funcs import MaxOnes, Sphere
from metaheuristic_designer.initializers import UniformInitializer
from metaheuristic_designer.operators.factories.mutation import create_mutation_operator
from metaheuristic_designer.survivor_selection import create_survivor_selection
from metaheuristic_designer.strategies import PopulationBasedStrategy


# ===================================================================
#  Random generator fixture
# ===================================================================
@pytest.fixture(scope="function")
def rng():
    """Return a Generator with seed 42. Use for reproducibility."""
    return np.random.default_rng(42)


# ===================================================================
#  Constants (genotype / fitness arrays)
# ===================================================================
SMALL_GENOTYPE = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
LARGE_GENOTYPE = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
EXAMPLE_FITNESS = np.array([-10, -2, -1, 0, 0, 1, 2, 10])
OFFSPRING_FITNESS_BETTER = np.array([-9, 10, 34, 2, 100, 2, 10, 100])
OFFSPRING_FITNESS_WORSE = np.array([-20, -5, -2, -1, -10, -90, -100, -10.1])
OFFSPRING_FITNESS_EQUAL = EXAMPLE_FITNESS.copy()
OFFSPRING_FITNESS_MIXED = np.array([-9, -5, 34, -1, 100, 2, 100, -10.1])
OFFSPRING_FITNESS_LOCAL_SEARCH = np.array(
    [
        -11,
        -3,
        2,
        -1,
        0,
        0,
        3,
        10,
        -9,
        4,
        1,
        -1,
        0,
        1,
        4,
        80,
        -1,
        -5,
        0,
        -1,
        0,
        0,
        5,
        80,
    ]
)
DE_POP = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
DE_FITNESS = np.array([0.1, 0.5, 0.3, 0.9])
DE_SMALL_POP = np.array([[0.0, 0.0], [1.0, 1.0]])


# ===================================================================
#  Dummy objective function and fixtures for simple wrappers
# ===================================================================
class DummyObjectiveFunction(ObjectiveFunc):
    """A stub that returns fixed fitness values and can track calls."""

    def __init__(
        self,
        name: str = "dummy",
        mode: str = "max",
        fitness_return=None,
        repair_return=None,
        **kwargs,
    ):
        super().__init__(mode=mode, name=name, **kwargs)
        self._fitness_return = fitness_return if fitness_return is not None else np.ones(1)
        self._repair_return = repair_return
        self.fitness_called = 0
        self.repair_called = 0

    def objective(self, solution: Any) -> np.ndarray:
        return self._fitness_return

    def calculate_fitness(
        self,
        population: Population,
    ) -> np.ndarray:
        self.fitness_called += 1
        if callable(self._fitness_return):
            fitness = self._fitness_return(population)
        else:
            fitness = self._fitness_return
        shape = (len(population.genotype_matrix),)
        fit_vector = np.broadcast_to(fitness, shape).copy()
        population.fitness = fit_vector
        population.objective = fit_vector
        population.best = np.atleast_2d(population.genotype_matrix[np.argmax(fit_vector)])
        population.best_objective = np.max(fit_vector)
        population.best_fitness = np.max(fit_vector)
        return population

    def repair_solution(self, population: Population) -> Population:
        self.repair_called += 1
        if self._repair_return is not None:
            repaired_matrix = self._repair_return(population.genotype_matrix)
            population.update_genotype(repaired_matrix)
        return population


@pytest.fixture
def dummy_objfunc():
    return DummyObjectiveFunction(dimension=3, lower_bound=0, upper_bound=1, name="dummy", mode="max")


@pytest.fixture
def dummy_objfunc_min():
    return DummyObjectiveFunction(dimension=3, name="dummy_min", mode="min")


# -------------------------------------------------------------------
#  Discrete and permutation objective classes (for simple wrapper tests)
# -------------------------------------------------------------------
class DiscreteSumObjective(ObjectiveFunc):
    def __init__(self, dimension=3, mode="max"):
        super().__init__(dimension, lower_bound=0, upper_bound=5, mode=mode, name="DiscreteSum")

    def objective(self, solution):
        return np.sum(solution)


class PermutationObjective(ObjectiveFunc):
    def __init__(self, dimension=4, mode="min"):
        super().__init__(dimension, lower_bound=0, upper_bound=dimension - 1, mode=mode, name="Permutation")

    def objective(self, solution):
        diff = np.abs(np.diff(solution))
        total = np.sum(diff) + np.abs(solution[0] - solution[-1])
        return total


@pytest.fixture
def binary_objfunc():
    return MaxOnes(dimension=5, mode="max")


@pytest.fixture
def discrete_objfunc():
    return DiscreteSumObjective(dimension=4, mode="max")


@pytest.fixture
def permutation_objfunc():
    return PermutationObjective(dimension=5, mode="min")


# ===================================================================
#  Dummy encodings
# ===================================================================
@pytest.fixture
def simple_encoding():
    """Identity encoding."""
    return DefaultEncoding()


class DummyParameterExtendingEncoding(ParameterExtendingEncoding):
    """Minimal parameter‑extending encoding for tests that need one."""

    def __init__(self, param_sizes):
        super().__init__(
            dimension=1,
            param_sizes=param_sizes,
            base_encoding=DefaultEncoding(),
        )


# ===================================================================
#  Population fixtures
# ===================================================================
@pytest.fixture
def empty_population():
    return Population(np.zeros((0, 2)))


@pytest.fixture
def example_population():
    """A 4‑individual population with pre-set fitness, historical best, etc."""
    pop = Population(np.arange(8).reshape(4, 2).astype(float))
    pop.fitness = np.array([3.0, 1.0, 4.0, 2.0])
    pop.historical_best_matrix = np.ones((4, 2))
    pop.historical_best_fitness = np.array([10.0, 20.0, 30.0, 40.0])
    pop.best = np.array([99.0, 99.0])
    pop.best_fitness = 99.0
    pop.best_objective = 99.0
    pop.fitness_calculated = np.array([True, False, True, False])
    return pop


# ===================================================================
#  Dummy operators and selections (trivial implementations)
# ===================================================================
@pytest.fixture
def dummy_operator():
    """Operator that returns the population unchanged."""
    return NullOperator()


@pytest.fixture
def dummy_parent_selection():
    """Parent selection that returns the population unmodified."""
    return NullParentSelection()


@pytest.fixture
def dummy_survivor_selection():
    """Survivor selection that keeps the offspring as the new population."""
    return NullSurvivorSelection()


# ===================================================================
#  Dummy initializer
# ===================================================================
@pytest.fixture
def dummy_initializer(rng):
    """Initializer that produces a population of 10 vectors of length 3 (uniform)."""
    return UniformInitializer(dimension=3, lower_bound=0, upper_bound=1, population_size=10, rng=rng)


# ===================================================================
#  Helper functions for expected random arrays (using seed 42)
# ===================================================================
def _expected_exponential(beta, size, seed=42):
    from metaheuristic_designer.initializers import ExponentialInitializer

    rng_fresh = np.random.default_rng(seed)
    init = ExponentialInitializer(size, beta, rng=rng_fresh)
    return init.generate_random()


def _expected_normal(mean, std, size, seed=42):
    from metaheuristic_designer.initializers import GaussianInitializer

    rng_fresh = np.random.default_rng(seed)
    init = GaussianInitializer(size, mean, std, rng=rng_fresh)
    return init.generate_random()


def _expected_uniform(low, high, size, seed=42):
    from metaheuristic_designer.initializers import UniformInitializer

    rng_fresh = np.random.default_rng(seed)
    init = UniformInitializer(size, low, high, rng=rng_fresh)
    return init.generate_random()


def _expected_permutation(n, seed=42):
    from metaheuristic_designer.initializers import PermInitializer

    rng_fresh = np.random.default_rng(seed)
    init = PermInitializer(10, n, rng=rng_fresh)
    return init.generate_random()


# ===================================================================
#  Helper: quick population with given fitness
# ===================================================================
def make_pop(fitness_list):
    """Return a Population with pre-set fitness, shape (len, 2)."""
    pop = Population(np.arange(len(fitness_list) * 2).reshape(len(fitness_list), 2).astype(float))
    pop.fitness = np.array(fitness_list)
    return pop


# ===================================================================
#  Fixtures for DE, permutation operator, and PSO tests
# ===================================================================
@pytest.fixture
def de_pop():
    return np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])


@pytest.fixture
def de_fitness():
    return np.array([0.1, 0.5, 0.3, 0.9])


@pytest.fixture
def perm_pop():
    """Small 4x4 integer matrix for permutation operator tests."""
    return np.array([[1, 2, 3, 4], [3, 4, 2, 1], [4, 1, 3, 2], [2, 3, 1, 4]])


@pytest.fixture
def pso_population():
    """A small population with PSOEncoding, fitness, historical best, and speed."""
    enc = PSOEncoding(dimension=2, base_encoding=DefaultEncoding())
    geno = np.array([[1.0, 2.0, 0.1, 0.2], [3.0, 4.0, 0.3, 0.4]])
    pop = Population(geno, encoding=enc)
    pop.fitness = np.array([0.5, 0.8])
    pop.historical_best_matrix = np.array([[1.0, 2.0, 0.1, 0.2], [3.0, 4.0, 0.3, 0.4]])
    pop.historical_best_fitness = np.array([0.5, 0.8])
    pop.best = np.array([3.0, 4.0, 0.3, 0.4])
    pop.best_fitness = 0.8
    pop.best_objective = 0.8
    return pop


@pytest.fixture
def sample_population_matrix():
    """4x3 float array for operator tests."""
    return np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])


# ===================================================================
#  Mock for sklearn GaussianProcessRegressor
# ===================================================================
class MockGaussianModel:
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def predict(self, X, return_std=False):
        mean = np.atleast_1d(self.mean)
        std = np.atleast_1d(self.std)
        if return_std:
            return mean, std
        return mean


# ===================================================================
#  Dummy search strategy fixture
# ===================================================================
@pytest.fixture
def dummy_strategy(dummy_initializer, dummy_operator, dummy_parent_selection, dummy_survivor_selection):
    return PopulationBasedStrategy(
        initializer=dummy_initializer,
        operator=dummy_operator,
        parent_sel=dummy_parent_selection,
        survivor_sel=dummy_survivor_selection,
        name="dummy_strategy",
    )


# ===================================================================
#  Sphere objective and simple integration strategy
# ===================================================================
@pytest.fixture
def sphere_objfunc():
    return Sphere(dimension=2, mode="min")


@pytest.fixture
def simple_strategy(rng):
    init = UniformInitializer(2, -10, 10, population_size=10, rng=rng)
    mut = create_mutation_operator("gauss", rng=rng, N=1, loc=0, scale=0.1)
    surv = create_survivor_selection("generational", rng=rng)
    return SearchStrategy(initializer=init, operator=mut, survivor_sel=surv, name="integration_strat")


# ===================================================================
#  Helper: run a simple wrapper and return best objective
# ===================================================================
def run_and_get_best(wrapper_func, objfunc, seed, **kwargs):
    """Run a simple wrapper for 5 generations and return the best objective."""
    run_kwargs = {
        "reporter": "silent",
        "stop_condition_str": "max_iterations",
        "max_iterations": 5,
        "max_evaluations": 1000,
        **kwargs,
    }
    algo = wrapper_func(objfunc, rng=seed, **run_kwargs)
    population = algo.optimize()
    _, best = population.best_solution()
    return best


# ===================================================================
#  ConfigurableHistoryTracker fixtures
# ===================================================================
@pytest.fixture
def full_tracker():
    """Tracker that records best, median, worst, and complete population."""
    return ConfigurableHistoryTracker(
        track_best=True,
        track_median=True,
        track_worst=True,
        track_full_population=True,
    )


@pytest.fixture
def algo_with_full_tracker(dummy_objfunc, dummy_strategy, full_tracker):
    """Algorithm with a pre-configured ConfigurableHistoryTracker and minimal stopping."""
    return Algorithm(
        dummy_objfunc,
        dummy_strategy,
        stop_condition_str="max_iterations",
        max_iterations=1,
        reporter="silent",
        history_tracker=full_tracker,
    )
