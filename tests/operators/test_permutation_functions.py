import numpy as np
from numpy.testing import assert_array_equal

from conftest import rng, perm_pop

from metaheuristic_designer.operators.operator_functions.permutation import (
    permute_mutation,
    roll_mutation,
    invert_mutation,
    pmx,
    order_cross,
)


# ===================================================================
#  permute_mutation
# ===================================================================
def test_permute_mutation_N2(rng, perm_pop):
    res = permute_mutation(perm_pop.copy(), None, random_state=rng, N=2)
    assert res.shape == perm_pop.shape

    # reproducibility
    rng2 = np.random.default_rng(42)
    exp = permute_mutation(perm_pop.copy(), None, random_state=rng2, N=2)
    assert_array_equal(res, exp)


def test_permute_mutation_N4(rng, perm_pop):
    res = permute_mutation(perm_pop.copy(), None, random_state=rng, N=4)
    assert res.shape == perm_pop.shape

    rng2 = np.random.default_rng(42)
    exp = permute_mutation(perm_pop.copy(), None, random_state=rng2, N=4)
    assert_array_equal(res, exp)


# ===================================================================
#  roll_mutation
# ===================================================================
def test_roll_mutation(rng, perm_pop):
    res = roll_mutation(perm_pop.copy(), None, random_state=rng, N=1)
    assert res.shape == perm_pop.shape

    rng2 = np.random.default_rng(42)
    exp = roll_mutation(perm_pop.copy(), None, random_state=rng2, N=1)
    assert_array_equal(res, exp)


# ===================================================================
#  invert_mutation
# ===================================================================
def test_invert_mutation(rng, perm_pop):
    res = invert_mutation(perm_pop.copy(), None, random_state=rng)
    assert res.shape == perm_pop.shape

    rng2 = np.random.default_rng(42)
    exp = invert_mutation(perm_pop.copy(), None, random_state=rng2)
    assert_array_equal(res, exp)


# ===================================================================
#  pmx
# ===================================================================
def test_pmx(rng, perm_pop):
    res = pmx(perm_pop.copy(), None, random_state=rng)
    assert res.shape == perm_pop.shape
    # pmx output is integer
    assert np.issubdtype(res.dtype, np.integer)

    rng2 = np.random.default_rng(42)
    exp = pmx(perm_pop.copy(), None, random_state=rng2)
    assert_array_equal(res, exp)


# ===================================================================
#  order_cross
# ===================================================================
def test_order_cross(rng, perm_pop):
    res = order_cross(perm_pop.copy(), None, random_state=rng)
    assert res.shape == perm_pop.shape
    assert np.issubdtype(res.dtype, np.integer)

    rng2 = np.random.default_rng(42)
    exp = order_cross(perm_pop.copy(), None, random_state=rng2)
    assert_array_equal(res, exp)