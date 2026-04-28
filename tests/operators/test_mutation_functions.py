import pytest
import numpy as np
from numpy.testing import assert_array_equal

# conftest fixtures
from conftest import rng

# functions under test
from metaheuristic_designer.operators.operator_functions.mutation import (
    xor_mask,
    mutate_noise,
    rand_noise,
    mutate_sample,
    rand_sample,
    sample_distribution,
    sample_1_sigma,
    mutate_1_sigma,
    mutate_n_sigmas,
    ProbDist,
)


# ===================================================================
#  xor_mask
# ===================================================================
def test_xor_mask_no_bits(rng):
    pop = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    result = xor_mask(pop, None, random_state=rng, N=0)
    assert_array_equal(result, pop)


def test_xor_mask_all_bytes(rng):
    pop = np.array([[0, 0], [0, 0]], dtype=np.uint8)
    result = xor_mask(pop, None, random_state=rng, N=2, BinRep="byte")
    assert result.shape == pop.shape
    # With N=2 and BinRep="byte", at least one element should differ
    assert np.any(result != 0)


def test_xor_mask_reproducible(rng):
    pop = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    res1 = xor_mask(pop, None, random_state=rng1, N=2, BinRep="byte")
    res2 = xor_mask(pop, None, random_state=rng2, N=2, BinRep="byte")
    assert_array_equal(res1, res2)


# ===================================================================
#  mutate_noise
# ===================================================================
def test_mutate_noise_no_change(rng):
    pop = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = mutate_noise(pop, None, random_state=rng, N=0, distrib=ProbDist.GAUSS, loc=0, scale=1)
    assert_array_equal(result, pop)


def test_mutate_noise_shape_and_reproducible(rng):
    pop = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = mutate_noise(pop, None, random_state=rng, N=2, distrib=ProbDist.GAUSS, loc=0, scale=1)
    assert result.shape == pop.shape

    rng2 = np.random.default_rng(42)
    expected = mutate_noise(np.array([[1.0, 2.0], [3.0, 4.0]]), None, random_state=rng2, N=2, distrib=ProbDist.GAUSS, loc=0, scale=1)
    assert_array_equal(result, expected)


# ===================================================================
#  rand_noise
# ===================================================================
def test_rand_noise_shape_and_reproducible(rng):
    pop = np.array([[0.0, 0.0], [0.0, 0.0]])
    result = rand_noise(pop, None, random_state=rng, distrib=ProbDist.GAUSS, loc=0, scale=1)
    assert result.shape == pop.shape

    rng2 = np.random.default_rng(42)
    expected = rand_noise(np.array([[0.0, 0.0], [0.0, 0.0]]), None, random_state=rng2, distrib=ProbDist.GAUSS, loc=0, scale=1)
    assert_array_equal(result, expected)


# ===================================================================
#  mutate_sample
# ===================================================================
def test_mutate_sample_no_change(rng):
    pop = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = mutate_sample(pop, None, random_state=rng, N=0, distrib=ProbDist.UNIFORM, loc=0, scale=1)
    assert_array_equal(result, pop)


def test_mutate_sample_shape_and_reproducible(rng):
    pop = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = mutate_sample(pop, None, random_state=rng, N=2, distrib=ProbDist.UNIFORM, loc=0, scale=1)
    assert result.shape == pop.shape

    rng2 = np.random.default_rng(42)
    expected = mutate_sample(np.array([[1.0, 2.0], [3.0, 4.0]]), None, random_state=rng2, N=2, distrib=ProbDist.UNIFORM, loc=0, scale=1)
    assert_array_equal(result, expected)


# ===================================================================
#  rand_sample
# ===================================================================
def test_rand_sample_shape_and_reproducible(rng):
    pop = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = rand_sample(pop, None, random_state=rng, distrib=ProbDist.UNIFORM, loc=0, scale=1)
    assert result.shape == pop.shape

    rng2 = np.random.default_rng(42)
    expected = rand_sample(np.array([[1.0, 2.0], [3.0, 4.0]]), None, random_state=rng2, distrib=ProbDist.UNIFORM, loc=0, scale=1)
    assert_array_equal(result, expected)


# ===================================================================
#  sample_distribution
# ===================================================================
def test_sample_distribution_gaussian(rng):
    shape = (2, 3)
    result = sample_distribution(shape, loc=0, scale=1, random_state=rng, distrib=ProbDist.GAUSS)
    assert result.shape == shape

    rng2 = np.random.default_rng(42)
    expected = sample_distribution(shape, loc=0, scale=1, random_state=rng2, distrib=ProbDist.GAUSS)
    assert_array_equal(result, expected)


def test_sample_distribution_uniform(rng):
    shape = (3, 2)
    result = sample_distribution(shape, loc=10, scale=5, random_state=rng, distrib=ProbDist.UNIFORM)
    assert result.shape == shape
    assert np.all(result >= 10) and np.all(result <= 15)


# ===================================================================
#  sample_1_sigma
# ===================================================================
def test_sample_1_sigma_shape_and_reproducible(rng):
    population = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    sigma = np.array([0.1, 0.1, 0.1])
    result = sample_1_sigma(population, None, random_state=rng, epsilon=0.01, sigma=sigma, tau=0.1, n=2)
    assert result.shape == population.shape

    rng2 = np.random.default_rng(42)
    expected = sample_1_sigma(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), None, random_state=rng2, epsilon=0.01, sigma=sigma, tau=0.1, n=2)
    assert_array_equal(result, expected)


# ===================================================================
#  mutate_1_sigma / mutate_n_sigmas
# ===================================================================
def test_mutate_1_sigma_shape_and_reproducible(rng):
    sigma = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = mutate_1_sigma(sigma, None, random_state=rng, epsilon=0.1, tau=0.5)
    assert result.shape == sigma.shape

    rng2 = np.random.default_rng(42)
    expected = mutate_1_sigma(np.array([[1.0, 2.0], [3.0, 4.0]]), None, random_state=rng2, epsilon=0.1, tau=0.5)
    assert_array_equal(result, expected)


def test_mutate_n_sigmas_shape_and_reproducible(rng):
    sigma = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = mutate_n_sigmas(sigma, None, random_state=rng, epsilon=0.1, tau=0.5, tau_multiple=0.2)
    assert result.shape == sigma.shape

    rng2 = np.random.default_rng(42)
    expected = mutate_n_sigmas(np.array([[1.0, 2.0], [3.0, 4.0]]), None, random_state=rng2, epsilon=0.1, tau=0.5, tau_multiple=0.2)
    assert_array_equal(result, expected)