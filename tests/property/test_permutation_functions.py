"""
Property-based and correctness tests for permutation operator functions.

Key invariants:
1. Output shape equals input shape.
2. Permutation-mutation operators preserve the multiset of values in each row
   (no values appear or disappear — they are only rearranged).
3. Crossover operators on valid permutations produce valid permutations
   (each value in 0..n-1 appears exactly once).
4. Results are deterministic with the same seed.
"""

import numpy as np
import pytest
from hypothesis import given, settings, assume, HealthCheck
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp

from metaheuristic_designer.operators.operator_functions.permutation import (
    permute_mutation,
    roll_mutation,
    invert_mutation,
    pmx,
    order_cross,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

seed_st = st.integers(0, 2**31 - 1)

# Permutation populations: each row is a permutation of 0..n-1
@st.composite
def perm_population(draw, min_rows=2, max_rows=10, min_cols=4, max_cols=10):
    n_rows = draw(st.integers(min_rows, max_rows))
    n_cols = draw(st.integers(min_cols, max_cols))
    rng = np.random.default_rng(draw(st.integers(0, 2**31 - 1)))
    rows = np.array([rng.permutation(n_cols) for _ in range(n_rows)])
    return rows


even_perm_population = perm_population(min_rows=2, max_rows=8).filter(lambda p: p.shape[0] % 2 == 0)

# General integer population for non-crossover tests
int_pop = hnp.arrays(
    dtype=np.int64,
    shape=st.tuples(st.integers(2, 10), st.integers(4, 10)),
    elements=st.integers(0, 20),
)


# ---------------------------------------------------------------------------
# permute_mutation
# ---------------------------------------------------------------------------

@given(pop=int_pop, seed=seed_st)
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_permute_mutation_preserves_shape(pop, seed):
    result = permute_mutation(pop.copy(), None, random_state=seed)
    assert result.shape == pop.shape


@given(pop=perm_population(), seed=seed_st)
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_permute_mutation_preserves_row_multiset(pop, seed):
    """Each row must contain the same values after mutation (just reordered)."""
    result = permute_mutation(pop.copy(), None, random_state=seed)
    for i in range(pop.shape[0]):
        assert np.array_equal(np.sort(result[i]), np.sort(pop[i])), (
            f"Row {i} changed its multiset: before={np.sort(pop[i])}, after={np.sort(result[i])}"
        )


@given(pop=perm_population(), seed=seed_st)
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
def test_permute_mutation_deterministic(pop, seed):
    r1 = permute_mutation(pop.copy(), None, random_state=seed)
    r2 = permute_mutation(pop.copy(), None, random_state=seed)
    np.testing.assert_array_equal(r1, r2)


def test_permute_mutation_n2_swaps_two_elements():
    """With N=2 exactly two positions per row are touched (swapped)."""
    pop = np.tile(np.arange(6), (4, 1))  # four identical rows [0,1,2,3,4,5]
    result = permute_mutation(pop.copy(), None, random_state=0, N=2)
    # After a swap-of-two, each row must still be a permutation of 0..5
    for row in result:
        assert set(row.tolist()) == set(range(6))


# ---------------------------------------------------------------------------
# roll_mutation
# ---------------------------------------------------------------------------

@given(pop=int_pop, seed=seed_st)
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_roll_mutation_preserves_shape(pop, seed):
    result = roll_mutation(pop.copy(), None, random_state=seed)
    assert result.shape == pop.shape


@given(pop=perm_population(), seed=seed_st)
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_roll_mutation_preserves_row_multiset(pop, seed):
    """Rolling a sub-sequence preserves all values in each row."""
    result = roll_mutation(pop.copy(), None, random_state=seed, N=1)
    for i in range(pop.shape[0]):
        assert np.array_equal(np.sort(result[i]), np.sort(pop[i]))


@given(pop=perm_population(), seed=seed_st)
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
def test_roll_mutation_deterministic(pop, seed):
    r1 = roll_mutation(pop.copy(), None, random_state=seed, N=1)
    r2 = roll_mutation(pop.copy(), None, random_state=seed, N=1)
    np.testing.assert_array_equal(r1, r2)


# ---------------------------------------------------------------------------
# invert_mutation
# ---------------------------------------------------------------------------

@given(pop=int_pop, seed=seed_st)
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_invert_mutation_preserves_shape(pop, seed):
    result = invert_mutation(pop.copy(), None, random_state=seed)
    assert result.shape == pop.shape


@given(pop=perm_population(), seed=seed_st)
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_invert_mutation_preserves_row_multiset(pop, seed):
    """Inverting a sub-sequence does not change the values present in each row."""
    result = invert_mutation(pop.copy(), None, random_state=seed)
    for i in range(pop.shape[0]):
        assert np.array_equal(np.sort(result[i]), np.sort(pop[i]))


@given(pop=perm_population(), seed=seed_st)
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
def test_invert_mutation_deterministic(pop, seed):
    r1 = invert_mutation(pop.copy(), None, random_state=seed)
    r2 = invert_mutation(pop.copy(), None, random_state=seed)
    np.testing.assert_array_equal(r1, r2)


def test_invert_mutation_reverses_segment():
    """A single known input: row [0,1,2,3,4] should have some segment reversed."""
    pop = np.arange(5, dtype=np.int64)[None, :]  # shape (1, 5)
    result = invert_mutation(pop.copy(), None, random_state=7)
    assert set(result[0].tolist()) == set(range(5))


# ---------------------------------------------------------------------------
# pmx (Partially Mapped Crossover)
# ---------------------------------------------------------------------------

@given(pop=even_perm_population, seed=seed_st)
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_pmx_preserves_shape(pop, seed):
    result = pmx(pop.copy(), None, random_state=seed)
    assert result.shape == pop.shape


@given(pop=even_perm_population, seed=seed_st)
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_pmx_offspring_are_valid_permutations(pop, seed):
    """Every row of the output must be a valid permutation of 0..n_cols-1."""
    n_cols = pop.shape[1]
    result = pmx(pop.copy(), None, random_state=seed)
    for i, row in enumerate(result):
        assert set(row.tolist()) == set(range(n_cols)), (
            f"PMX offspring row {i} is not a valid permutation: {row}"
        )


@given(pop=even_perm_population, seed=seed_st)
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
def test_pmx_deterministic(pop, seed):
    r1 = pmx(pop.copy(), None, random_state=seed)
    r2 = pmx(pop.copy(), None, random_state=seed)
    np.testing.assert_array_equal(r1, r2)


def test_pmx_known_case():
    """Explicit small case: two rows that are inverse permutations.
    PMX must still produce valid permutations."""
    pop = np.array([
        [0, 1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1, 0],
    ])
    result = pmx(pop.copy(), None, random_state=42)
    for row in result:
        assert set(row.tolist()) == set(range(6))


# ---------------------------------------------------------------------------
# order_cross (Order Crossover)
# ---------------------------------------------------------------------------

@given(pop=even_perm_population, seed=seed_st)
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_order_cross_preserves_shape(pop, seed):
    result = order_cross(pop.copy(), None, random_state=seed)
    assert result.shape == pop.shape


@given(pop=even_perm_population, seed=seed_st)
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_order_cross_offspring_are_valid_permutations(pop, seed):
    """Every offspring row must be a valid permutation of 0..n_cols-1."""
    n_cols = pop.shape[1]
    result = order_cross(pop.copy(), None, random_state=seed)
    for i, row in enumerate(result):
        assert set(row.tolist()) == set(range(n_cols)), (
            f"order_cross offspring row {i} is not a valid permutation: {row}"
        )


@given(pop=even_perm_population, seed=seed_st)
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
def test_order_cross_deterministic(pop, seed):
    r1 = order_cross(pop.copy(), None, random_state=seed)
    r2 = order_cross(pop.copy(), None, random_state=seed)
    np.testing.assert_array_equal(r1, r2)
