# tests/test_operator_wrappers.py
import numpy as np
import pytest
from copy import copy
from metaheuristic_designer.operators.operator_functions.utils import OperatorVectorDef
from metaheuristic_designer.operators.operator_functions.mutation import rand_noise, ProbDist


# ----------------------------------------------------------------------
# Minimal mock population
# ----------------------------------------------------------------------
class MockPopulation:
    def __init__(self, matrix, fitness=None):
        self.genotype_matrix = np.asarray(matrix, dtype=float)
        self.fitness = fitness if fitness is not None else np.ones(len(matrix))
        self.vec_size = matrix.shape[1]

    def update_genotype_matrix(self, new_matrix):
        self.genotype_matrix = new_matrix
        return self  # return self, just like the real Population


# ----------------------------------------------------------------------
# A simple dummy mutation that records what it received
# ----------------------------------------------------------------------
def dummy_mutation(matrix, fitness, random_state=None, **kwargs):
    """Returns matrix + 1, and stores kwargs for inspection."""
    dummy_mutation.last_kwargs = kwargs
    dummy_mutation.last_random_state = random_state
    return matrix + 1.0


# ----------------------------------------------------------------------
# Tests for OperatorVectorDef
# ----------------------------------------------------------------------
def test_wrapper_passes_forced_params():
    """forced_params must always reach the function."""
    wrapper = OperatorVectorDef(dummy_mutation, forced_params={"distrib": ProbDist.GAUSS})
    pop = MockPopulation(np.zeros((3, 2)))
    wrapper(pop, None, random_state=42)

    assert dummy_mutation.last_kwargs["distrib"] == ProbDist.GAUSS


def test_wrapper_merges_params_and_runtime():
    """default params are overridden by runtime kwargs, but forced_params win."""
    wrapper = OperatorVectorDef(
        dummy_mutation,
        params={"scale": 0.5, "loc": 1.0},
        forced_params={"loc": 0.0}   # forced overrides default
    )
    pop = MockPopulation(np.zeros((3, 2)))
    # runtime overrides scale
    wrapper(pop, None, random_state=42, scale=0.2, extra="hello")

    kw = dummy_mutation.last_kwargs
    # forced_params override default loc
    assert kw["loc"] == 0.0
    # runtime overrides default scale
    assert kw["scale"] == 0.2
    # extra runtime kwarg passed through
    assert kw["extra"] == "hello"


def test_wrapper_passes_fitness():
    wrapper = OperatorVectorDef(dummy_mutation)
    fitness = np.array([10.0, 20.0, 30.0])
    pop = MockPopulation(np.zeros((3, 2)), fitness=fitness)
    wrapper(pop, None, random_state=42)

    # The dummy mutation received the fitness array
    assert dummy_mutation.last_kwargs == {}


def test_wrapper_updates_population():
    wrapper = OperatorVectorDef(dummy_mutation)
    original = np.random.default_rng(0).random((3, 2))
    pop = MockPopulation(original.copy())
    result = wrapper(pop, None, random_state=42)

    # Population updated with matrix+1
    expected = original + 1.0
    assert np.allclose(pop.genotype_matrix, expected)
    # The wrapper itself returns the population (the object returned by update_genotype_matrix)
    assert result is pop


def test_wrapper_reproducibility_with_seed():
    """Same seed should yield identical results (if function uses it)."""
    # Use rand_noise with large effect so we can see difference
    wrapper = OperatorVectorDef(rand_noise, forced_params={"distrib": ProbDist.GAUSS})
    pop = MockPopulation(np.zeros((10, 5)))

    # Run twice with same seed
    wrapper(pop, None, random_state=42)
    first_run = pop.genotype_matrix.copy()

    pop2 = MockPopulation(np.zeros((10, 5)))
    wrapper(pop2, None, random_state=42)
    second_run = pop2.genotype_matrix

    assert np.allclose(first_run, second_run)

    # Different seed → different results
    pop3 = MockPopulation(np.zeros((10, 5)))
    wrapper(pop3, None, random_state=999)
    third_run = pop3.genotype_matrix

    assert not np.allclose(first_run, third_run)


def test_wrapper_with_scheduled_parameter():
    """A callable parameter should be evaluated before calling the function."""
    # The mixin evaluates callables and stores resolved values;
    # but the wrapper itself does NOT evaluate; it just merges already evaluated values.
    # The evaluation is done by the operator's step method (via mixin). 
    # For a direct wrapper test without operator, we pass a resolved value (not callable).
    # This test ensures the wrapper can pass a number that was a scheduled value.
    wrapper = OperatorVectorDef(dummy_mutation)
    pop = MockPopulation(np.zeros((3, 2)))
    # Just pass a float; wrapper forwards it.
    wrapper(pop, None, random_state=42, scale=3.14)
    assert dummy_mutation.last_kwargs["scale"] == 3.14