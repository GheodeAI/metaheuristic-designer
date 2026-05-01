"""
Shared fixtures and mocks for all package tests.
No classes are defined inside test modules; everything lives here.
"""

import numpy as np
import pytest
from copy import copy
from typing import Optional, Iterable, Any, Tuple, List

# -------------------------------------------------------------------
#  Adjust imports to your actual package layout
# -------------------------------------------------------------------
from metaheuristic_designer.encoding import (
    Encoding,
    DefaultEncoding,
    EncodingFromLambda,
)
from metaheuristic_designer.encodings.parameter_extending_encoding import (
    ParameterExtendingEncoding,
)
from metaheuristic_designer.population import Population
from metaheuristic_designer.objective_function import (
    ObjectiveFunc,
    VectorObjectiveFunc,
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
from metaheuristic_designer.utils import check_random_state
from metaheuristic_designer.encodings import PSOEncoding  # real encoding with speed param
from metaheuristic_designer.population import Population

# ===================================================================
#  Fixed‑seed random generator fixture
# ===================================================================
@pytest.fixture(scope="function")
def rng():
    """Return a Generator with seed 42. Use for reproducibility."""
    return np.random.default_rng(42)


# ===================================================================
#  Shared genotype / fitness constants
# ===================================================================
SMALL_GENOTYPE = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
LARGE_GENOTYPE = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
EXAMPLE_FITNESS = np.array([-10, -2, -1, 0, 0, 1, 2, 10])
OFFSPRING_FITNESS_BETTER = np.array([-9, 10, 34, 2, 100, 2, 10, 100])
OFFSPRING_FITNESS_WORSE = np.array([-20, -5, -2, -1, -10, -90, -100, -10.1])
OFFSPRING_FITNESS_EQUAL = EXAMPLE_FITNESS.copy()
OFFSPRING_FITNESS_MIXED = np.array([-9, -5, 34, -1, 100, 2, 100, -10.1])
OFFSPRING_FITNESS_LOCAL_SEARCH = np.array([
    -11, -3,  2, -1,  0,  0,  3, 10,
     -9,  4,  1, -1,  0,  1,  4, 80,
     -1, -5,  0, -1,  0,  0,  5, 80,
])
# DE test data
DE_POP = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
DE_FITNESS = np.array([0.1, 0.5, 0.3, 0.9])
DE_SMALL_POP = np.array([[0.0, 0.0], [1.0, 1.0]])


# ===================================================================
#  Dummy objective function (configurable)
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
        # Pass minimal required arguments to ABC
        super().__init__(mode=mode, name=name, **kwargs)
        self._fitness_return = (
            fitness_return if fitness_return is not None else np.ones(1)
        )
        self._repair_return = repair_return
        self.fitness_called = 0
        self.repair_called = 0

    def objective(self, solution: Any) -> np.ndarray:
        raise NotImplementedError("Use fitness() directly in tests")

    def fitness(
        self,
        population: Population,
        adjusted: bool = False,
        parallel: bool = False,
        threads: int = 8,
    ) -> np.ndarray:
        self.fitness_called += 1
        if callable(self._fitness_return):
            return self._fitness_return(population)
        shape = (len(population.genotype_matrix),)
        return np.broadcast_to(self._fitness_return, shape).copy()

    def repair_solution(self, solution: np.ndarray) -> np.ndarray:
        self.repair_called += 1
        if self._repair_return is not None:
            return self._repair_return(solution)
        return solution


@pytest.fixture
def dummy_objfunc():
    return DummyObjectiveFunction(name="dummy", mode="max")


@pytest.fixture
def dummy_objfunc_min():
    return DummyObjectiveFunction(name="dummy_min", mode="min")


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
            vecsize=1,                  # not used by most tests
            param_sizes=param_sizes,
            base_encoding=DefaultEncoding(),
        )


# ===================================================================
#  Population fixtures
# ===================================================================
@pytest.fixture
def empty_population(dummy_objfunc):
    return Population(dummy_objfunc, np.zeros((0, 2)))


@pytest.fixture
def example_population(dummy_objfunc):
    """A 4‑individual population with pre‑set fitness, historical best, etc."""
    pop = Population(dummy_objfunc, np.arange(8).reshape(4, 2).astype(float))
    pop.fitness = np.array([3.0, 1.0, 4.0, 2.0])
    pop.historical_best_matrix = np.ones((4, 2))
    pop.historical_best_fitness = np.array([10.0, 20.0, 30.0, 40.0])
    pop.best = np.array([99.0, 99.0])
    pop.best_fitness = 99.0
    pop.fitness_calculated = np.array([True, False, True, False])
    return pop


# ===================================================================
#  Dummy operators, selections (trivial implementations)
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
#  Dummy initializer (a simple fixed‑size initializer)
# ===================================================================
@pytest.fixture
def dummy_initializer(rng):
    """Initializer that produces a population of 10 vectors of length 3 (uniform)."""
    from metaheuristic_designer.initializers import UniformInitializer
    return UniformInitializer(genotype_size=3, low_lim=0, up_lim=1, pop_size=10, random_state=rng)


# ===================================================================
#  Helper: expected random arrays for key initialisers (using seed 42)
# ===================================================================
def _expected_exponential(beta, size, seed=42):
    from metaheuristic_designer.initializers import ExponentialInitializer
    rng_fresh = np.random.default_rng(seed)
    init = ExponentialInitializer(size, beta, random_state=rng_fresh)
    return init.generate_random()

def _expected_normal(mean, std, size, seed=42):
    from metaheuristic_designer.initializers import GaussianInitializer
    rng_fresh = np.random.default_rng(seed)
    init = GaussianInitializer(size, mean, std, random_state=rng_fresh)
    return init.generate_random()

def _expected_uniform(low, high, size, seed=42):
    from metaheuristic_designer.initializers import UniformInitializer
    rng_fresh = np.random.default_rng(seed)
    init = UniformInitializer(size, low, high, random_state=rng_fresh)
    return init.generate_random()

def _expected_permutation(n, seed=42):
    from metaheuristic_designer.initializers import PermInitializer
    rng_fresh = np.random.default_rng(seed)
    init = PermInitializer(n, random_state=rng_fresh)
    return init.generate_random()

# Helper to quickly create a Population with given fitness
def make_pop(fitness_list, objfunc):
    """Return a Population with pre‑set fitness, shape (len, 2)."""
    pop = Population(objfunc, np.arange(len(fitness_list) * 2).reshape(len(fitness_list), 2).astype(float))
    pop.fitness = np.array(fitness_list)
    return pop

# Small arrays for DE operator tests
@pytest.fixture
def de_pop():
    return np.array([[0.0, 1.0],
                     [2.0, 3.0],
                     [4.0, 5.0],
                     [6.0, 7.0]])

@pytest.fixture
def de_fitness():
    return np.array([0.1, 0.5, 0.3, 0.9])   # best is index 3

@pytest.fixture
def perm_pop():
    """Small 4x4 integer matrix for permutation operator tests."""
    return np.array([[1, 2, 3, 4],
                     [3, 4, 2, 1],
                     [4, 1, 3, 2],
                     [2, 3, 1, 4]])

@pytest.fixture
def pso_population(dummy_objfunc):
    """A small population with PSOEncoding, fitness, historical best, and speed."""
    enc = PSOEncoding(vecsize=2, base_encoding=DefaultEncoding())
    geno = np.array([[1.0, 2.0, 0.1, 0.2],   # solution (1,2) + speed (0.1,0.2)
                     [3.0, 4.0, 0.3, 0.4]])
    pop = Population(dummy_objfunc, geno, encoding=enc)
    pop.fitness = np.array([0.5, 0.8])
    pop.historical_best_matrix = np.array([[1.0, 2.0, 0.1, 0.2],
                                           [3.0, 4.0, 0.3, 0.4]])
    pop.historical_best_fitness = np.array([0.5, 0.8])
    pop.best = np.array([3.0, 4.0, 0.3, 0.4])  # best solution with its speed
    pop.best_fitness = 0.8
    return pop

@pytest.fixture
def sample_population_matrix():
    """4x3 float array for operator tests."""
    return np.array([[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0],
                     [7.0, 8.0, 9.0],
                     [10.0, 11.0, 12.0]])

class MockGaussianModel:
    """Mock for sklearn GaussianProcessRegressor."""
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def predict(self, X, return_std=False):
        mean = np.atleast_1d(self.mean)
        std = np.atleast_1d(self.std)
        if return_std:
            return mean, std
        return mean

@pytest.fixture
def dummy_strategy(dummy_initializer, dummy_operator, dummy_parent_selection, dummy_survivor_selection):
    from metaheuristic_designer.search_strategy import SearchStrategy
    return SearchStrategy(
        initializer=dummy_initializer,
        operator=dummy_operator,
        parent_sel=dummy_parent_selection,
        survivor_sel=dummy_survivor_selection,
        name="dummy_strategy",
    )

from metaheuristic_designer import SearchStrategy
from metaheuristic_designer.benchmarks.benchmark_funcs import Sphere
from metaheuristic_designer.initializers import UniformInitializer
from metaheuristic_designer.operators.mutation_operator import create_mutation_operator
from metaheuristic_designer.survivor_selection import create_survivor_selection

@pytest.fixture
def sphere_objfunc():
    return Sphere(vecsize=2, mode="min")

@pytest.fixture
def simple_strategy(sphere_objfunc, rng):
    # Initializer: small population, 2D, bounds [-10,10]
    init = UniformInitializer(2, -10, 10, pop_size=10, random_state=rng)
    # Operator: Gaussian noise mutation with small noise
    mut = create_mutation_operator("gauss", random_state=rng, N=1, loc=0, scale=0.1)
    # Survivor selection: generational (replace parents with offspring)
    surv = create_survivor_selection("generational", random_state=rng)
    strat = SearchStrategy(initializer=init, operator=mut, survivor_sel=surv, name="integration_strat")
    return strat