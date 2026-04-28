import pytest
import numpy as np
from numpy.testing import assert_array_equal

# conftest fixture
from conftest import rng

# functions under test
from metaheuristic_designer.operators.operator_functions.crossover import (
    one_point_crossover,
    two_point_crossover,
    uniform_crossover,
    multiparent_discrete_crossover,
    averaged_crossover,
    blx_alpha_crossover,
    sbx_crossover,
    bitwise_xor_crossover,
    cross_inter_avg,
)


# ===================================================================
#  one_point_crossover
# ===================================================================
def test_one_point_crossover_shape_and_reproducible(rng):
    pop = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9,10,11,12],
                    [13,14,15,16]])
    result = one_point_crossover(pop, None, random_state=rng)
    assert result.shape == pop.shape

    rng2 = np.random.default_rng(42)
    expected = one_point_crossover(pop, None, random_state=rng2)
    assert_array_equal(result, expected)


def test_one_point_crossover_single_gene(rng):
    # shape[1] == 1 -> returns unchanged
    pop = np.array([[1], [2]])
    result = one_point_crossover(pop, None, random_state=rng)
    assert_array_equal(result, pop)


# ===================================================================
#  two_point_crossover
# ===================================================================
def test_two_point_crossover_shape_and_reproducible(rng):
    pop = np.array([[1,2,3,4,5],
                    [6,7,8,9,10],
                    [11,12,13,14,15],
                    [16,17,18,19,20]])
    result = two_point_crossover(pop, None, random_state=rng)
    assert result.shape == pop.shape

    rng2 = np.random.default_rng(42)
    expected = two_point_crossover(pop, None, random_state=rng2)
    assert_array_equal(result, expected)


# ===================================================================
#  uniform_crossover
# ===================================================================
def test_uniform_crossover_shape_and_reproducible(rng):
    pop = np.arange(12).reshape(3, 4)
    result = uniform_crossover(pop, None, random_state=rng)
    assert result.shape == pop.shape

    rng2 = np.random.default_rng(42)
    expected = uniform_crossover(pop, None, random_state=rng2)
    assert_array_equal(result, expected)


# ===================================================================
#  multiparent_discrete_crossover
# ===================================================================
def test_multiparent_discrete_crossover_shape_and_reproducible(rng):
    pop = np.arange(15).reshape(5, 3)
    result = multiparent_discrete_crossover(pop, None, N=3, random_state=rng)
    assert result.shape == pop.shape

    rng2 = np.random.default_rng(42)
    expected = multiparent_discrete_crossover(pop, None, N=3, random_state=rng2)
    assert_array_equal(result, expected)


# ===================================================================
#  averaged_crossover
# ===================================================================
def test_averaged_crossover_shape_and_reproducible(rng):
    pop = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = averaged_crossover(pop, None, alpha=0.5, random_state=rng)
    assert result.shape == pop.shape

    rng2 = np.random.default_rng(42)
    expected = averaged_crossover(pop, None, alpha=0.5, random_state=rng2)
    assert_array_equal(result, expected)


# ===================================================================
#  blx_alpha_crossover
# ===================================================================
def test_blx_alpha_crossover_shape_and_reproducible(rng):
    pop = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = blx_alpha_crossover(pop, None, alpha=0.5, low=0.0, high=10.0, random_state=rng)
    assert result.shape == pop.shape

    rng2 = np.random.default_rng(42)
    expected = blx_alpha_crossover(pop, None, alpha=0.5, low=0.0, high=10.0, random_state=rng2)
    assert_array_equal(result, expected)


# ===================================================================
#  sbx_crossover
# ===================================================================
def test_sbx_crossover_shape_and_reproducible(rng):
    pop = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = sbx_crossover(pop, None, F=1.0, random_state=rng)
    assert result.shape == pop.shape

    rng2 = np.random.default_rng(42)
    expected = sbx_crossover(pop, None, F=1.0, random_state=rng2)
    assert_array_equal(result, expected)


# ===================================================================
#  bitwise_xor_crossover
# ===================================================================
def test_bitwise_xor_crossover_shape_and_reproducible(rng):
    pop = np.array([[0, 255], [128, 64], [32, 16]], dtype=np.uint8)
    result = bitwise_xor_crossover(pop, None, random_state=rng)
    assert result.shape == pop.shape

    rng2 = np.random.default_rng(42)
    expected = bitwise_xor_crossover(pop, None, random_state=rng2)
    assert_array_equal(result, expected)


# ===================================================================
#  cross_inter_avg
# ===================================================================
def test_cross_inter_avg_shape_and_reproducible(rng):
    pop = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = cross_inter_avg(pop, None, N=2, random_state=rng)
    assert result.shape == pop.shape

    rng2 = np.random.default_rng(42)
    expected = cross_inter_avg(pop, None, N=2, random_state=rng2)
    assert_array_equal(result, expected)


def test_cross_inter_avg_single_parent_returns_self(rng):
    # N=1 should average with itself -> no change (but implementation adds population itself N times and divides by N)
    pop = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = cross_inter_avg(pop, None, N=1, random_state=rng)
    assert_array_equal(result, pop)   # (pop + pop)/1 = pop


def test_cross_inter_avg_does_not_mutate_input(rng):
    # Known bug: cross_inter_avg currently modifies its input array in place.
    # This test will fail until the function is fixed.
    pop = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    original = pop.copy()
    cross_inter_avg(pop, None, N=2, random_state=rng)
    assert_array_equal(pop, original)   # will fail because pop is mutated


def test_cross_inter_avg_N_equal_pop_size(rng):
    pop = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = cross_inter_avg(pop.copy(), None, N=2, random_state=rng)
    assert result.shape == pop.shape


def test_cross_inter_avg_N_larger_than_pop(rng):
    # N is clipped to pop size, so this should work
    pop = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = cross_inter_avg(pop.copy(), None, N=5, random_state=rng)
    assert result.shape == pop.shape


# Additional edge-case tests for other crossovers

def test_averaged_crossover_alpha_zero(rng):
    # alpha=0 -> no change
    pop = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = averaged_crossover(pop.copy(), None, alpha=0.0, random_state=rng)
    assert_array_equal(result, pop)


def test_sbx_crossover_single_pair(rng):
    # With only two individuals, it should still work
    pop = np.array([[0.0, 0.0], [1.0, 1.0]])
    result = sbx_crossover(pop.copy(), None, F=1.0, random_state=rng)
    assert result.shape == pop.shape


def test_one_point_crossover_odd_population(rng):
    # odd number of individuals
    pop = np.array([[1,2,3], [4,5,6], [7,8,9]])
    result = one_point_crossover(pop.copy(), None, random_state=rng)
    assert result.shape == pop.shape