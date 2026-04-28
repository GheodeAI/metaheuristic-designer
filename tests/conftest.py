import pytest
import numpy as np

# Adjust imports to your project layout
from metaheuristic_designer.encoding import DefaultEncoding
from metaheuristic_designer.population import Population


# -------------------------------------------------------------------
#  Dummy objective function (unchanged from before)
# -------------------------------------------------------------------
class DummyObjectiveFunction:
    """Minimal stub of ObjectiveFunc."""
    def __init__(self, name="dummy", mode="max", fitness_return=None, repair_return=None):
        self.name = name
        self.mode = mode
        self._fitness_return = fitness_return if fitness_return is not None else np.ones(1)
        self._repair_return = repair_return
        self.fitness_called = 0
        self.repair_called = 0

    def fitness(self, population, *, parallel=False, threads=8):
        self.fitness_called += 1
        if callable(self._fitness_return):
            return self._fitness_return(population)
        shape = (len(population.genotype_matrix),)
        return np.broadcast_to(self._fitness_return, shape).copy()

    def repair_solution(self, individual):
        self.repair_called += 1
        if self._repair_return is not None:
            return self._repair_return(individual)
        return individual


# -------------------------------------------------------------------
#  Shared constants for Population tests (still here)
# -------------------------------------------------------------------
SMALL_GENOTYPE = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
LARGE_GENOTYPE = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
GENERIC_FITNESS = np.array([-10.0, 0.0, 10.0])


# -------------------------------------------------------------------
#  Fixtures for Population tests
# -------------------------------------------------------------------
@pytest.fixture
def simple_encoding():
    return DefaultEncoding()

@pytest.fixture
def dummy_objfunc():
    return DummyObjectiveFunction(name="dummy", mode="max")

@pytest.fixture
def dummy_objfunc_min():
    return DummyObjectiveFunction(name="dummy_min", mode="min")

@pytest.fixture
def empty_population(dummy_objfunc):
    return Population(dummy_objfunc, np.zeros((0, 2)))


# -------------------------------------------------------------------
#  New fixtures for initializer tests
# -------------------------------------------------------------------
@pytest.fixture(scope="function")
def rng():
    """Fixed‑seed Generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def objfunc():
    """A lightweight objective function for initializer tests."""
    return DummyObjectiveFunction(name="test_obj", mode="max")


@pytest.fixture
def encoding():
    """Default (identity) encoding."""
    return DefaultEncoding()