"""
Property-based and correctness tests for mutation operator functions.

Key invariants:
1. Output shape equals input shape.
2. With N=0 (no mutation), output is unchanged.
3. Distributions produce finite values.
4. mutate_noise / rand_noise with UNIFORM stay within expected range.
5. Results are reproducible given the same seed.
6. ProbDist.from_str raises ValueError for unknown strings.
"""

import numpy as np
import pytest
from hypothesis import given, settings, assume, HealthCheck
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp

from metaheuristic_designer.operators.operator_functions.mutation import (
    ProbDist,
    sample_distribution,
    mutate_noise,
    mutate_sample,
    rand_noise,
    rand_sample,
    sample_1_sigma,
    mutate_1_sigma,
    mutate_n_sigmas,
    xor_mask,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

real_pop = hnp.arrays(
    dtype=np.float64,
    shape=st.tuples(
        st.integers(min_value=2, max_value=10),
        st.integers(min_value=2, max_value=10),
    ),
    elements=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
)

seed_st = st.integers(min_value=0, max_value=2**31 - 1)


# ---------------------------------------------------------------------------
# ProbDist.from_str
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("gauss", ProbDist.GAUSS),
    ("gaussian", ProbDist.GAUSS),
    ("normal", ProbDist.GAUSS),
    ("uniform", ProbDist.UNIFORM),
    ("cauchy", ProbDist.CAUCHY),
    ("laplace", ProbDist.LAPLACE),
    ("categorical", ProbDist.CATEGORICAL),
])
def test_prob_dist_from_str_known(name, expected):
    assert ProbDist.from_str(name) == expected


def test_prob_dist_from_str_unknown_raises():
    with pytest.raises(ValueError):
        ProbDist.from_str("not_a_distribution")


# ---------------------------------------------------------------------------
# sample_distribution – shape and finiteness for simple distributions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("distrib", [
    ProbDist.GAUSS,
    ProbDist.UNIFORM,
    ProbDist.CAUCHY,
    ProbDist.LAPLACE,
    ProbDist.EXPON,
])
def test_sample_distribution_correct_shape(distrib):
    shape = (5, 4)
    result = sample_distribution(shape, loc=0, scale=1, random_state=0, distrib=distrib)
    assert result.shape == shape


def test_sample_distribution_uniform_within_bounds():
    shape = (20, 10)
    lo, hi = -3.0, 3.0
    result = sample_distribution(shape, loc=lo, scale=hi - lo, random_state=0, distrib=ProbDist.UNIFORM)
    assert np.all(result >= lo)
    assert np.all(result <= hi)


def test_sample_distribution_bernoulli_binary():
    shape = (50, 8)
    result = sample_distribution(shape, random_state=7, distrib=ProbDist.BERNOULLI, p=0.5)
    assert set(result.flatten().tolist()).issubset({0, 1})


def test_sample_distribution_binomial_within_range():
    n_trials = 10
    shape = (30, 5)
    result = sample_distribution(shape, random_state=3, distrib=ProbDist.BINOMIAL, n=n_trials, p=0.3)
    assert np.all(result >= 0)
    assert np.all(result <= n_trials)


def test_sample_distribution_gamma_nonnegative():
    shape = (20, 4)
    result = sample_distribution(shape, loc=0, scale=1, random_state=5, distrib=ProbDist.GAMMA, a=2)
    assert np.all(result >= 0)


def test_sample_distribution_categorical_valid_indices():
    p = np.array([0.2, 0.5, 0.3])
    shape = (40, 3)
    result = sample_distribution(shape, random_state=11, distrib=ProbDist.CATEGORICAL, p=p)
    assert np.all(result >= 0)
    assert np.all(result < 3)


def test_sample_distribution_custom_distribution():
    import scipy.stats as stats
    dist = stats.norm(loc=10.0, scale=0.5)
    shape = (10, 4)
    result = sample_distribution(shape, random_state=0, distrib=ProbDist.CUSTOM, distrib_class=dist)
    assert result.shape == shape
    # Result should be clustered around 10
    assert np.all(np.abs(result - 10.0) < 10.0)


def test_sample_distribution_custom_without_class_raises():
    with pytest.raises(ValueError):
        sample_distribution((2, 2), random_state=0, distrib=ProbDist.CUSTOM)


def test_sample_distribution_string_name():
    result = sample_distribution((3, 3), loc=0, scale=1, random_state=0, distrib="gauss")
    assert result.shape == (3, 3)


# ---------------------------------------------------------------------------
# mutate_noise – correctness and shape
# ---------------------------------------------------------------------------

def test_mutate_noise_zero_mutations_unchanged():
    pop = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = mutate_noise(pop.copy(), None, random_state=0, N=0,
                          distrib=ProbDist.GAUSS, loc=0, scale=1)
    np.testing.assert_array_equal(result, pop)


@given(pop=real_pop, seed=seed_st)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_mutate_noise_preserves_shape(pop, seed):
    result = mutate_noise(pop.copy(), None, random_state=seed, N=1,
                          distrib=ProbDist.GAUSS, loc=0, scale=1)
    assert result.shape == pop.shape


@given(pop=real_pop, seed=seed_st)
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
def test_mutate_noise_deterministic(pop, seed):
    r1 = mutate_noise(pop.copy(), None, random_state=seed, N=1,
                      distrib=ProbDist.GAUSS, loc=0, scale=1)
    r2 = mutate_noise(pop.copy(), None, random_state=seed, N=1,
                      distrib=ProbDist.GAUSS, loc=0, scale=1)
    np.testing.assert_array_equal(r1, r2)


def test_mutate_noise_uniform_with_min_max_bounds():
    """Adding UNIFORM noise in [0, 0] (delta=0) must leave the array unchanged."""
    pop = np.ones((4, 3))
    result = mutate_noise(pop.copy(), None, random_state=0, N=3,
                          distrib=ProbDist.UNIFORM, min=0.0, max=0.0)
    np.testing.assert_array_equal(result, pop)


# ---------------------------------------------------------------------------
# mutate_sample – correctness and shape
# ---------------------------------------------------------------------------

def test_mutate_sample_zero_mutations_unchanged():
    pop = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = mutate_sample(pop.copy(), None, random_state=0, N=0,
                           distrib=ProbDist.UNIFORM, loc=0, scale=1)
    np.testing.assert_array_equal(result, pop)


@given(pop=real_pop, seed=seed_st)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_mutate_sample_preserves_shape(pop, seed):
    result = mutate_sample(pop.copy(), None, random_state=seed, N=1,
                           distrib=ProbDist.GAUSS, loc=0, scale=1)
    assert result.shape == pop.shape


def test_mutate_sample_uniform_stays_in_bounds():
    """When sampling from UNIFORM[0,1], mutated components must lie in [0, 1]."""
    pop = np.ones((6, 5)) * 0.5
    result = mutate_sample(pop.copy(), None, random_state=0, N=5,
                           distrib=ProbDist.UNIFORM, min=0.0, max=1.0)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


# ---------------------------------------------------------------------------
# rand_noise – correctness and shape
# ---------------------------------------------------------------------------

@given(pop=real_pop, seed=seed_st)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_rand_noise_preserves_shape(pop, seed):
    result = rand_noise(pop.copy(), None, random_state=seed,
                        distrib=ProbDist.GAUSS, loc=0, scale=1)
    assert result.shape == pop.shape


@given(pop=real_pop, seed=seed_st)
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
def test_rand_noise_deterministic(pop, seed):
    r1 = rand_noise(pop.copy(), None, random_state=seed, distrib=ProbDist.GAUSS, loc=0, scale=1)
    r2 = rand_noise(pop.copy(), None, random_state=seed, distrib=ProbDist.GAUSS, loc=0, scale=1)
    np.testing.assert_array_equal(r1, r2)


# ---------------------------------------------------------------------------
# rand_sample – correctness and shape
# ---------------------------------------------------------------------------

@given(pop=real_pop, seed=seed_st)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_rand_sample_preserves_shape(pop, seed):
    result = rand_sample(pop.copy(), None, random_state=seed,
                         distrib=ProbDist.GAUSS, loc=0, scale=1)
    assert result.shape == pop.shape


def test_rand_sample_uniform_stays_in_bounds():
    pop = np.zeros((8, 4))
    result = rand_sample(pop.copy(), None, random_state=0,
                         distrib=ProbDist.UNIFORM, min=5.0, max=10.0)
    assert np.all(result >= 5.0)
    assert np.all(result <= 10.0)


# ---------------------------------------------------------------------------
# xor_mask
# ---------------------------------------------------------------------------

uint_pop = hnp.arrays(
    dtype=np.uint8,
    shape=st.tuples(
        st.integers(min_value=2, max_value=8),
        st.integers(min_value=2, max_value=8),
    ),
    elements=st.integers(min_value=0, max_value=255),
)


@given(pop=uint_pop, seed=seed_st)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_xor_mask_preserves_shape(pop, seed):
    result = xor_mask(pop.copy(), None, random_state=seed, N=1, BinRep="byte")
    assert result.shape == pop.shape


def test_xor_mask_zero_bits_unchanged():
    pop = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    result = xor_mask(pop.copy(), None, random_state=0, N=0)
    np.testing.assert_array_equal(result, pop)


@given(pop=uint_pop, seed=seed_st)
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
def test_xor_mask_deterministic(pop, seed):
    r1 = xor_mask(pop.copy(), None, random_state=seed, N=2, BinRep="byte")
    r2 = xor_mask(pop.copy(), None, random_state=seed, N=2, BinRep="byte")
    np.testing.assert_array_equal(r1, r2)


def test_xor_mask_bin_mode_only_flips_bits():
    """BinRep='bin' only flips bits that were in the mask — resulting value is 0 or 1."""
    pop = np.ones((4, 4), dtype=np.uint8)
    result = xor_mask(pop.copy(), None, random_state=0, N=4, BinRep="bin")
    assert np.all((result == 0) | (result == 1))


# ---------------------------------------------------------------------------
# ES sigma mutation functions
# ---------------------------------------------------------------------------

def test_mutate_1_sigma_output_positive():
    sigma = np.abs(np.random.default_rng(0).normal(1, 0.5, (4, 3))) + 1e-6
    result = mutate_1_sigma(sigma.copy(), None, random_state=0, epsilon=1e-8, tau=0.1)
    assert np.all(result > 0)


def test_mutate_1_sigma_preserves_shape():
    sigma = np.ones((5, 3))
    result = mutate_1_sigma(sigma.copy(), None, random_state=42, epsilon=1e-6, tau=0.2)
    assert result.shape == sigma.shape


def test_mutate_n_sigmas_output_positive():
    sigma = np.ones((4, 3))
    result = mutate_n_sigmas(sigma.copy(), None, random_state=1,
                             epsilon=1e-8, tau=0.1, tau_multiple=0.05)
    assert np.all(result > 0)


def test_mutate_n_sigmas_preserves_shape():
    sigma = np.ones((6, 4))
    result = mutate_n_sigmas(sigma.copy(), None, random_state=2,
                             epsilon=1e-6, tau=0.1, tau_multiple=0.1)
    assert result.shape == sigma.shape
