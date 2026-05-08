"""
Shared fixtures for the full test suite.

Rules:
- One fixture per logical entity; no setup duplication across files.
- Fixtures are minimal and focused; they do not hide bugs.
- All random state uses explicit seeds so tests are deterministic.
"""

import numpy as np
import pytest

from metaheuristic_designer.objective_function import ObjectiveFunc, ObjectiveFromLambda
from metaheuristic_designer.initializers import UniformInitializer
from metaheuristic_designer.benchmarks.benchmark_funcs import MaxOnes, Sphere
from metaheuristic_designer.population import Population
from metaheuristic_designer.operator import NullOperator
from metaheuristic_designer.parent_selection_base import NullParentSelection
from metaheuristic_designer.survivor_selection_base import NullSurvivorSelection


# ---------------------------------------------------------------------------
# Random state
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    """A seeded Generator used everywhere a deterministic random source is needed."""
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Objective functions
# ---------------------------------------------------------------------------

@pytest.fixture
def onemax_func():
    """MaxOnes objective on 8 binary variables (maximisation)."""
    return MaxOnes(dimension=8)


@pytest.fixture
def sphere_min():
    """Sphere minimisation on 5 real variables in [-5, 5]."""
    return Sphere(dimension=5, mode="min")


@pytest.fixture
def simple_max_func():
    """Trivial lambda objective: fitness = sum of vector (maximisation)."""
    return ObjectiveFromLambda(lambda x: float(x.sum()), mode="max", name="sum")


# ---------------------------------------------------------------------------
# Initialisers
# ---------------------------------------------------------------------------

@pytest.fixture
def uniform_initializer():
    """Uniform initialiser over [0, 1]^4 with a population of 6."""
    return UniformInitializer(
        dimension=4,
        lower_bound=np.zeros(4),
        upper_bound=np.ones(4),
        pop_size=6,
    )


@pytest.fixture
def binary_initializer():
    """Uniform initialiser over {0,1}^8 with a population of 10."""
    return UniformInitializer(
        dimension=8,
        lower_bound=np.zeros(8),
        upper_bound=np.ones(8),
        pop_size=10,
    )


# ---------------------------------------------------------------------------
# Populations (pre-built, no fitness evaluated)
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_population(onemax_func):
    """A Population of 6 individuals, 8 binary genes, fitness NOT calculated."""
    rng = np.random.default_rng(0)
    geno = rng.integers(0, 2, size=(6, 8)).astype(float)
    return Population(onemax_func, geno)


@pytest.fixture
def evaluated_population(onemax_func):
    """A Population of 6 individuals with fitness already calculated."""
    rng = np.random.default_rng(0)
    geno = rng.integers(0, 2, size=(6, 8)).astype(float)
    pop = Population(onemax_func, geno)
    pop.calculate_fitness()
    return pop


@pytest.fixture
def real_population(sphere_min):
    """A Population of 8 individuals on the Sphere function, fitness calculated."""
    rng = np.random.default_rng(1)
    geno = rng.uniform(-5, 5, size=(8, 5))
    pop = Population(sphere_min, geno)
    pop.calculate_fitness()
    return pop


# ---------------------------------------------------------------------------
# Fitness arrays used directly by selection function tests
# ---------------------------------------------------------------------------

@pytest.fixture
def fitness_asc():
    """Fitness values in ascending order: [1, 2, 3, 4, 5]."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def fitness_mixed():
    """Fitness with a clear best (index 2, value 10) and a clear worst (index 4, value -1)."""
    return np.array([3.0, 1.0, 10.0, 4.0, -1.0])
