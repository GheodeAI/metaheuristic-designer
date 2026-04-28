import pytest
import numpy as np
from numpy.testing import assert_array_equal

from conftest import rng, dummy_initializer, simple_encoding, make_pop, dummy_objfunc

from metaheuristic_designer.operators.operator_functions.random_generation import (
    compute_statistic,
    random_initialize,
    random_reset,
)
from metaheuristic_designer.operators.random_operator import create_random_operator
from metaheuristic_designer.operator import OperatorFromLambda
from metaheuristic_designer.population import Population


# ===================================================================
#  compute_statistic (unchanged)
# ===================================================================
def test_compute_statistic_mean():
    arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = compute_statistic(arr, "mean")
    expected = np.array([3.0, 4.0])
    assert_array_equal(result, expected)


def test_compute_statistic_median():
    arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = compute_statistic(arr, "median")
    expected = np.array([3.0, 4.0])
    assert_array_equal(result, expected)


def test_compute_statistic_std():
    arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = compute_statistic(arr, "std")
    expected = np.std(arr, axis=0)
    assert_array_equal(result, expected)


def test_compute_statistic_average_weighted():
    arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    weights = np.array([0.5, 0.5, 0.0])
    result = compute_statistic(arr, "average", weights=weights)
    expected = np.average(arr, axis=0, weights=weights)
    assert_array_equal(result, expected)


# ===================================================================
#  random_initialize
# ===================================================================
def test_random_initialize_shape_and_modification(rng, dummy_initializer):
    arr = np.zeros((5, 3))               # 3 columns to match dummy_initializer
    result = random_initialize(arr, dummy_initializer)
    assert result.shape == arr.shape
    assert not np.array_equal(result, arr)


def test_random_initialize_reproducible(rng, dummy_initializer):
    arr = np.zeros((3, 3))               # 3 columns
    res1 = random_initialize(arr, dummy_initializer)
    dummy_initializer.random_state = np.random.default_rng(42)
    res2 = random_initialize(arr, dummy_initializer)
    assert_array_equal(res1, res2)


# ===================================================================
#  random_reset
# ===================================================================
def test_random_reset_shape_and_modification(rng, dummy_initializer):
    arr = np.ones((4, 3))                # 3 columns
    result = random_reset(arr, dummy_initializer, random_state=rng, n=2)
    assert result.shape == arr.shape
    assert not np.array_equal(result, arr)


def test_random_reset_reproducible(rng, dummy_initializer):
    arr = np.ones((3, 3))                # 3 columns
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    res1 = random_reset(arr.copy(), dummy_initializer, random_state=rng1, n=2)
    res2 = random_reset(arr.copy(), dummy_initializer, random_state=rng2, n=2)
    assert_array_equal(res1, res2)


# ===================================================================
#  Factory: create_random_operator
# ===================================================================
@pytest.mark.parametrize("method", ["random", "reset"])
def test_create_random_operator_returns_operator(method, rng, simple_encoding):
    op = create_random_operator(method, encoding=simple_encoding, random_state=rng)
    assert isinstance(op, OperatorFromLambda)
    assert op.name == method


def test_create_random_operator_invalid_method():
    with pytest.raises(KeyError):
        create_random_operator("invalid_rando")


# -------------------------------------------------------------------
#  Integration: operator modifies population via factory
# -------------------------------------------------------------------
def test_random_operator_full_reset(rng, dummy_objfunc, simple_encoding, dummy_initializer):
    # Build a population with 3 columns to match dummy_initializer
    geno = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    pop = Population(dummy_objfunc, geno, encoding=simple_encoding)
    pop.fitness = np.zeros(2)
    original = pop.genotype_matrix.copy()

    op = create_random_operator("random", encoding=simple_encoding, random_state=rng)
    result = op(pop, initializer=dummy_initializer)
    assert result is pop
    assert not np.array_equal(pop.genotype_matrix, original)