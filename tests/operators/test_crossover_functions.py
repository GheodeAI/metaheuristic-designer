import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

# conftest fixture
from conftest import rng

# Updated functions under test
from metaheuristic_designer.operators.operator_functions.crossover import (
    k_point_crossover,
    uniform_crossover,
    averaged_crossover,
    blend_crossover,
    sbx_crossover,
    bitwise_xor_crossover,
    multiparent_discrete_crossover,
    multiparent_intermediate_crossover,
)


# ===================================================================
#  k_point_crossover (replaces one‑point and two‑point)
# ===================================================================
def test_k_point_crossover_shape_even(rng):
    pop = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    result = k_point_crossover(pop, None, k=1, random_state=rng)
    assert result.shape == pop.shape


def test_k_point_crossover_shape_odd(rng):
    pop = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    result = k_point_crossover(pop, None, k=2, random_state=rng)
    assert result.shape == pop.shape


def test_k_point_crossover_reproducible(rng):
    pop = np.arange(12).reshape(3, 4)
    rng2 = np.random.default_rng(42)
    res1 = k_point_crossover(pop, None, k=1, random_state=rng)
    res2 = k_point_crossover(pop, None, k=1, random_state=rng2)
    assert_array_equal(res1, res2)


def test_k_point_crossover_prob_zero(rng):
    # crossover_prob = 0 -> offspring identical to parents
    pop = np.arange(12).reshape(3, 4)
    result = k_point_crossover(pop, None, k=1, crossover_prob=0.0, random_state=rng)
    # All rows of result should be present in pop (possibly reordered)
    for row in result:
        assert any(np.allclose(row, p) for p in pop)


def test_k_point_crossover_prob_one(rng):
    pop = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    result = k_point_crossover(pop, None, k=1, crossover_prob=1.0, random_state=rng)
    # With prob=1, all pairs must be crossed; offspring should not equal parents
    # At least one pair should be crossed (may be same if k=0 but k=1 guarantees cut)
    assert not np.array_equal(result, pop)


# ===================================================================
#  uniform_crossover
# ===================================================================
def test_uniform_crossover_shape_even(rng):
    pop = np.arange(12).reshape(3, 4)
    result = uniform_crossover(pop, None, random_state=rng)
    assert result.shape == pop.shape


def test_uniform_crossover_shape_odd(rng):
    pop = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    result = uniform_crossover(pop, None, random_state=rng)
    assert result.shape == pop.shape


def test_uniform_crossover_reproducible(rng):
    pop = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    rng2 = np.random.default_rng(42)
    res1 = uniform_crossover(pop, None, random_state=rng)
    res2 = uniform_crossover(pop, None, random_state=rng2)
    assert_array_equal(res1, res2)


def test_uniform_crossover_prob_zero(rng):
    pop = np.arange(12).reshape(3, 4)
    result = uniform_crossover(pop, None, crossover_prob=0.0, random_state=rng)
    for row in result:
        assert any(np.allclose(row, p) for p in pop)


def test_uniform_crossover_prob_one(rng):
    pop = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    result = uniform_crossover(pop, None, crossover_prob=1.0, random_state=rng)
    # With prob=1, all pairs are crossed; result should differ from original (unless all genes identical, unlikely)
    assert not np.array_equal(result, pop)


# ===================================================================
#  averaged_crossover (arithmetic)
# ===================================================================
def test_averaged_crossover_shape_even(rng):
    pop = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    result = averaged_crossover(pop, None, alpha=0.5, random_state=rng)
    assert result.shape == pop.shape


def test_averaged_crossover_shape_odd(rng):
    pop = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = averaged_crossover(pop, None, alpha=0.3, random_state=rng)
    assert result.shape == pop.shape


def test_averaged_crossover_prob_zero(rng):
    pop = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = averaged_crossover(pop, None, alpha=0.5, crossover_prob=0.0, pairing_method="stable", random_state=rng)
    # With no crossover, offspring are copies of parents (possibly reordered)
    for row in result:
        assert any(np.allclose(row, p) for p in pop)


def test_averaged_crossover_alpha_zero(rng):
    pop = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    # alpha=0 -> children are exact copies of parents (but cross order may swap)
    # Because when cross_prob=1, child1 = parents1, child2 = parents2 (blend)
    result = averaged_crossover(pop, None, alpha=0.0, crossover_prob=1.0, pairing_method="stable", random_state=rng)
    # The concatenated children are exactly the same multiset as parents
    assert set(map(tuple, result)) == set(map(tuple, pop))


def test_averaged_crossover_alpha_half(rng):
    pop = np.array([[0.0, 0.0], [2.0, 2.0], [4.0, 4.0], [6.0, 6.0]])
    result = averaged_crossover(pop, None, alpha=0.5, crossover_prob=1.0, random_state=rng)
    # Offspring values should be averages of pairs
    # Since pairing is random, it's hard to predict exact values, but we can check bounds
    assert np.all(result <= pop.max())
    assert np.all(result >= pop.min())


# ===================================================================
#  blend_crossover (BLX-alpha)
# ===================================================================
def test_blend_crossover_shape_even(rng):
    pop = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    result = blend_crossover(pop, None, alpha=0.5, random_state=rng)
    assert result.shape == pop.shape


def test_blend_crossover_shape_odd(rng):
    pop = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = blend_crossover(pop, None, alpha=0.5, random_state=rng)
    assert result.shape == pop.shape


def test_blend_crossover_bounds(rng):
    pop = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    result = blend_crossover(pop, None, alpha=0.5, crossover_prob=1.0, random_state=rng)
    # With alpha=0.5, the interval expands by 0.5*range. For a pair (a,b) with a<b,
    # lower = a - 0.5*(b-a) = 1.5a - 0.5b, upper = 1.5b - 0.5a.
    # Overall min possible across pairs is -0.5*max_range, max = max + 0.5*max_range.
    # So values should be within [pop.min() - 0.5*(pop.max()-pop.min()), pop.max() + ...]
    low = pop.min()
    high = pop.max()
    rng_val = high - low
    bound_low = low - 0.5 * rng_val
    bound_high = high + 0.5 * rng_val
    assert np.all(result >= bound_low)
    assert np.all(result <= bound_high)


def test_blend_crossover_prob_zero(rng):
    pop = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = blend_crossover(pop, None, alpha=0.5, crossover_prob=0.0, pairing_method="stable", random_state=rng)
    # Offspring should be exactly the parents (multiset preserved)
    for row in result:
        assert any(np.allclose(row, p) for p in pop)


# ===================================================================
#  sbx_crossover
# ===================================================================
def test_sbx_crossover_shape_even(rng):
    pop = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    result = sbx_crossover(pop, None, eta=0.5, random_state=rng)
    assert result.shape == pop.shape


def test_sbx_crossover_shape_odd(rng):
    pop = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    result = sbx_crossover(pop, None, eta=1.0, random_state=rng)
    assert result.shape == pop.shape


def test_sbx_crossover_prob_zero(rng):
    pop = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    result = sbx_crossover(pop, None, eta=2.0, crossover_prob=0.0, pairing_method="stable", random_state=rng)
    # Offspring is exactly the parents (unsorted but multiset identical)
    for row in result:
        assert any(np.allclose(row, p) for p in pop)


def test_sbx_crossover_symmetry(rng):
    # With eta large, children should be near parents
    pop = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [2.0, 2.0]])
    result = sbx_crossover(pop, None, eta=20.0, crossover_prob=1.0, random_state=rng)
    # Children should not exceed the parent range too far (moderate spread)
    # They should be within [min-epsilon, max+epsilon]
    assert np.all(result >= -0.5) and np.all(result <= 3.0)  # or even use pop.min() - 0.5, pop.max() + 0.5


# ===================================================================
#  bitwise_xor_crossover
# ===================================================================
def test_bitwise_xor_shape_boolean(rng):
    pop = np.array([[True, False], [False, True], [True, True], [False, False]])
    result = bitwise_xor_crossover(pop, None, random_state=rng)
    assert result.shape == pop.shape
    assert result.dtype == np.bool_


def test_bitwise_xor_shape_integer(rng):
    pop = np.array([[0, 255], [128, 64], [32, 16]], dtype=np.uint8)
    result = bitwise_xor_crossover(pop, None, random_state=rng)
    assert result.shape == pop.shape


def test_bitwise_xor_prob_zero(rng):
    pop = np.array([[1, 0], [0, 1], [1, 1]])
    result = bitwise_xor_crossover(pop, None, crossover_prob=0.0, pairing_method="stable", random_state=rng)
    for row in result:
        assert any(np.allclose(row, p) for p in pop)


def test_bitwise_xor_prob_one_bool(rng):
    pop = np.array([[True, False], [False, True], [True, True], [False, False]])
    result = bitwise_xor_crossover(pop, None, crossover_prob=1.0, random_state=rng)
    # For booleans, child1 = p1^p2, child2 = p1^(~p2) which is the complement of child1.
    # So if any pair has different parents, offspring will differ.
    # We can't guarantee they differ because parents might be equal.
    # Just check shape.
    assert result.shape == pop.shape


# ===================================================================
#  multiparent_discrete_crossover
# ===================================================================
def test_multiparent_discrete_shape(rng):
    pop = np.arange(15).reshape(5, 3)
    result = multiparent_discrete_crossover(pop, None, k=3, random_state=rng)
    assert result.shape == pop.shape


def test_multiparent_discrete_prob_zero(rng):
    pop = np.arange(15).reshape(5, 3)
    result = multiparent_discrete_crossover(pop, None, k=3, crossover_prob=0.0, random_state=rng)
    assert_array_equal(result, pop)  # no crossover, exact copy


def test_multiparent_discrete_prob_one(rng):
    pop = np.arange(15).reshape(5, 3)
    result = multiparent_discrete_crossover(pop, None, k=3, crossover_prob=1.0, random_state=rng)
    # Crossed offspring; each gene is drawn from the selected parents.
    # The result should not be identical to pop (unless very small pop with identical values, unlikely)
    assert not np.array_equal(result, pop)


def test_multiparent_discrete_replace_false(rng):
    pop = np.arange(12).reshape(4, 3)
    # With replace=False, the three parents for each offspring must be distinct
    # Not directly testable without inspecting internal indices, but we trust the implementation.
    # We test that no rows contain values outside the pop's range.
    result = multiparent_discrete_crossover(pop, None, k=3, replace=False, random_state=rng)
    assert np.all(np.isin(result, pop))


def test_multiparent_discrete_replace_true(rng):
    pop = np.arange(12).reshape(4, 3)
    result = multiparent_discrete_crossover(pop, None, k=3, replace=True, random_state=rng)
    assert np.all(np.isin(result, pop))


# ===================================================================
#  multiparent_intermediate_crossover
# ===================================================================
def test_multiparent_intermediate_shape(rng):
    pop = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    result = multiparent_intermediate_crossover(pop, None, k=3, random_state=rng)
    assert result.shape == pop.shape


def test_multiparent_intermediate_prob_zero(rng):
    pop = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = multiparent_intermediate_crossover(pop, None, k=2, crossover_prob=0.0, random_state=rng)
    assert_array_equal(result, pop)


def test_multiparent_intermediate_prob_one(rng):
    pop = np.array([[0.0, 0.0], [2.0, 2.0], [4.0, 4.0], [6.0, 6.0]])
    result = multiparent_intermediate_crossover(pop, None, k=4, crossover_prob=1.0, random_state=rng)
    # With k=N, all offspring are the population mean (0+2+4+6)/4 = 3.0
    # Since replace=False and k=4 exactly the whole pop, each offspring is mean of whole pop.
    expected_mean = np.full_like(pop, 3.0)
    assert_allclose(result, expected_mean, atol=1e-12)


def test_multiparent_intermediate_fixed_seed(rng):
    pop = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    rng2 = np.random.default_rng(42)
    res1 = multiparent_intermediate_crossover(pop, None, k=2, random_state=rng)
    res2 = multiparent_intermediate_crossover(pop, None, k=2, random_state=rng2)
    assert_array_equal(res1, res2)
