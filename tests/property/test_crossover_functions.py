"""
Property-based tests for crossover operator functions.

Key invariants tested with Hypothesis:
1. Output shape equals input shape (dimensionality preserved).
2. Each gene in the offspring comes from one of the two parents (no new values invented).
3. Crossover is deterministic given the same seed.
4. averaged_crossover keeps values within the convex hull of the two parents
   (component-wise, since it's a weighted average).
"""

import numpy as np
import pytest
from hypothesis import given, settings, assume, HealthCheck
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp

from metaheuristic_designer.operators.operator_functions.crossover import (
    one_point_crossover,
    two_point_crossover,
    uniform_crossover,
    multiparent_discrete_crossover,
    averaged_crossover,
    blx_alpha_crossover,
    sbx_crossover,
    bitwise_xor_crossover,
)

# A strategy for populations: even number of rows (≥4), at least 2 columns.
even_population = hnp.arrays(
    dtype=np.float64,
    shape=st.tuples(
        st.integers(min_value=2, max_value=12).map(lambda x: x * 2),
        st.integers(min_value=2, max_value=12),
    ),
    elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
)

seed_st = st.integers(min_value=0, max_value=2**31 - 1)


# ---------------------------------------------------------------------------
# one_point_crossover
# ---------------------------------------------------------------------------

@given(pop=even_population, seed=seed_st)
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_one_point_crossover_preserves_shape(pop, seed):
    result = one_point_crossover(pop.copy(), None, random_state=seed)
    assert result.shape == pop.shape


@given(pop=even_population, seed=seed_st)
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_one_point_crossover_each_gene_from_a_parent(pop, seed):
    """Each gene in the offspring must come from one of the two parent pools."""
    result = one_point_crossover(pop.copy(), None, random_state=seed)
    n = pop.shape[0]
    half = n // 2
    parents1 = pop[:half]
    parents2 = pop[half:]
    for j in range(half):
        for col in range(pop.shape[1]):
            assert result[j, col] in (parents1[j, col], parents2[j, col]), (
                f"offspring[{j},{col}]={result[j,col]} not from parents"
            )


@given(pop=even_population, seed=seed_st)
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
def test_one_point_crossover_deterministic(pop, seed):
    r1 = one_point_crossover(pop.copy(), None, random_state=seed)
    r2 = one_point_crossover(pop.copy(), None, random_state=seed)
    np.testing.assert_array_equal(r1, r2)


# ---------------------------------------------------------------------------
# two_point_crossover  (requires ≥ 3 columns)
# ---------------------------------------------------------------------------

wide_even_population = hnp.arrays(
    dtype=np.float64,
    shape=st.tuples(
        st.integers(min_value=2, max_value=8).map(lambda x: x * 2),
        st.integers(min_value=3, max_value=12),
    ),
    elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
)


@given(pop=wide_even_population, seed=seed_st)
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_two_point_crossover_preserves_shape(pop, seed):
    result = two_point_crossover(pop.copy(), None, random_state=seed)
    assert result.shape == pop.shape


@given(pop=wide_even_population, seed=seed_st)
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_two_point_crossover_each_gene_from_a_parent(pop, seed):
    result = two_point_crossover(pop.copy(), None, random_state=seed)
    n = pop.shape[0]
    half = n // 2
    parents1 = pop[:half]
    parents2 = pop[half:]
    for j in range(half):
        for col in range(pop.shape[1]):
            assert result[j, col] in (parents1[j, col], parents2[j, col])


# ---------------------------------------------------------------------------
# uniform_crossover
# ---------------------------------------------------------------------------

@given(pop=even_population, seed=seed_st)
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_uniform_crossover_preserves_shape(pop, seed):
    result = uniform_crossover(pop.copy(), None, random_state=seed)
    assert result.shape == pop.shape


@given(pop=even_population, seed=seed_st)
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_uniform_crossover_each_gene_from_a_parent(pop, seed):
    result = uniform_crossover(pop.copy(), None, random_state=seed)
    n = pop.shape[0]
    half = n // 2
    parents1 = pop[:half]
    parents2 = pop[half:]
    for j in range(half):
        for col in range(pop.shape[1]):
            assert result[j, col] in (parents1[j, col], parents2[j, col])


@given(pop=even_population, seed=seed_st)
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
def test_uniform_crossover_deterministic(pop, seed):
    r1 = uniform_crossover(pop.copy(), None, random_state=seed)
    r2 = uniform_crossover(pop.copy(), None, random_state=seed)
    np.testing.assert_array_equal(r1, r2)


# ---------------------------------------------------------------------------
# averaged_crossover: output in convex hull of the two original populations
# ---------------------------------------------------------------------------

@given(pop=even_population, alpha=st.floats(0.0, 1.0, allow_nan=False), seed=seed_st)
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_averaged_crossover_preserves_shape(pop, alpha, seed):
    result = averaged_crossover(pop.copy(), None, alpha=alpha, random_state=seed)
    assert result.shape == pop.shape


@given(pop=even_population, seed=seed_st)
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_averaged_crossover_output_within_input_range(pop, seed):
    """averaged_crossover is a weighted average, so each component
    must lie within [min_col, max_col] of the original population."""
    result = averaged_crossover(pop.copy(), None, alpha=0.5, random_state=seed)
    col_min = pop.min(axis=0)
    col_max = pop.max(axis=0)
    assert np.all(result >= col_min - 1e-9)
    assert np.all(result <= col_max + 1e-9)


# ---------------------------------------------------------------------------
# blx_alpha_crossover
# ---------------------------------------------------------------------------

@given(pop=even_population, alpha=st.floats(0.0, 1.0, allow_nan=False), seed=seed_st)
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_blx_alpha_crossover_preserves_shape(pop, alpha, seed):
    result = blx_alpha_crossover(pop.copy(), None, alpha=alpha, random_state=seed)
    assert result.shape == pop.shape


# ---------------------------------------------------------------------------
# bitwise_xor_crossover (integer populations)
# ---------------------------------------------------------------------------

int_population = hnp.arrays(
    dtype=np.int32,
    shape=st.tuples(
        st.integers(min_value=2, max_value=10).map(lambda x: x * 2),
        st.integers(min_value=2, max_value=10),
    ),
    elements=st.integers(min_value=0, max_value=255),
)


@given(pop=int_population, seed=seed_st)
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_bitwise_xor_crossover_preserves_shape(pop, seed):
    result = bitwise_xor_crossover(pop.copy(), None, random_state=seed)
    assert result.shape == pop.shape
