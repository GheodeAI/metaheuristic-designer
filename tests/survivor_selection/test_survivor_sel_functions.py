import pytest
import numpy as np
from numpy.testing import assert_array_equal

# Conftest constants
from conftest import (
    EXAMPLE_FITNESS,
    OFFSPRING_FITNESS_BETTER,
    OFFSPRING_FITNESS_WORSE,
    OFFSPRING_FITNESS_EQUAL,
    OFFSPRING_FITNESS_MIXED,
    OFFSPRING_FITNESS_LOCAL_SEARCH,
    rng,
)

# Functions under test
from metaheuristic_designer.survivor_selection.survivor_selection_functions import (
    generational,
    one_to_one,
    prob_one_to_one,
    many_to_one,
    prob_many_to_one,
    elitism,
    cond_elitism,
    keep_best,
    keep_best_offspring,
)


# ===================================================================
#  generational
# ===================================================================
def test_generational():
    result = generational(EXAMPLE_FITNESS, OFFSPRING_FITNESS_BETTER, None)
    expected = np.arange(8, 16)
    assert_array_equal(result, expected)


# ===================================================================
#  one_to_one
# ===================================================================
@pytest.mark.parametrize("population_fitness", [EXAMPLE_FITNESS])
@pytest.mark.parametrize(
    "offspring_fitness, expected",
    [
        (OFFSPRING_FITNESS_BETTER, np.array([8, 9, 10, 11, 12, 13, 14, 15])),
        (OFFSPRING_FITNESS_WORSE, np.array([0, 1, 2, 3, 4, 5, 6, 7])),
        (OFFSPRING_FITNESS_EQUAL, np.array([8, 9, 10, 11, 12, 13, 14, 15])),
        (OFFSPRING_FITNESS_MIXED, np.array([8, 1, 10, 3, 12, 13, 14, 7])),
    ],
)
def test_one_to_one(population_fitness, offspring_fitness, expected):
    result = one_to_one(population_fitness, offspring_fitness, None)
    assert result.max() < len(population_fitness) + len(offspring_fitness)
    assert result.min() >= 0
    assert len(result) == len(population_fitness)
    assert_array_equal(result, expected)


# ===================================================================
#  prob_one_to_one (deterministic for p = 0 or 1)
# ===================================================================
@pytest.mark.parametrize("population_fitness", [EXAMPLE_FITNESS])
@pytest.mark.parametrize(
    "offspring_fitness, p, expected",
    [
        (OFFSPRING_FITNESS_BETTER, 0.0, np.array([8, 9, 10, 11, 12, 13, 14, 15])),
        (OFFSPRING_FITNESS_WORSE, 0.0, np.array([0, 1, 2, 3, 4, 5, 6, 7])),
        (OFFSPRING_FITNESS_BETTER, 1.0, np.array([8, 9, 10, 11, 12, 13, 14, 15])),
        (OFFSPRING_FITNESS_WORSE, 1.0, np.array([8, 9, 10, 11, 12, 13, 14, 15])),
    ],
)
def test_prob_one_to_one_extremes(population_fitness, offspring_fitness, p, expected, rng):
    # With p=0, only strictly better offspring replace; with p=1, all replace.
    result = prob_one_to_one(population_fitness, offspring_fitness, rng, p)
    assert result.max() < 16
    assert result.min() >= 0
    assert len(result) == 8
    assert_array_equal(result, expected)


@pytest.mark.parametrize("population_fitness", [EXAMPLE_FITNESS])
@pytest.mark.parametrize(
    "offspring_fitness",
    [
        OFFSPRING_FITNESS_BETTER,
        OFFSPRING_FITNESS_WORSE,
        OFFSPRING_FITNESS_EQUAL,
        OFFSPRING_FITNESS_MIXED,
    ],
)
@pytest.mark.parametrize("p", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_prob_one_to_one_shape(population_fitness, offspring_fitness, p, rng):
    result = prob_one_to_one(population_fitness, offspring_fitness, rng, p)
    assert result.max() < 16
    assert result.min() >= 0
    assert len(result) == 8


# ===================================================================
#  many_to_one
# ===================================================================
@pytest.mark.parametrize(
    "parent_fitness, offspring_fitness, expected",
    [
        (EXAMPLE_FITNESS, OFFSPRING_FITNESS_LOCAL_SEARCH, np.arange(8) + 8 * np.array([3, 2, 1, 0, 0, 0, 3, 2])),
    ],
)
def test_many_to_one(parent_fitness, offspring_fitness, expected):
    result = many_to_one(parent_fitness, offspring_fitness, None)
    assert result.max() < 32
    assert result.min() >= 0
    assert len(result) == 8
    assert_array_equal(result, expected)


# ===================================================================
#  prob_many_to_one (extremes are deterministic)
# ===================================================================
@pytest.mark.parametrize("parent_fitness", [EXAMPLE_FITNESS])
@pytest.mark.parametrize("p", [0.0, 1.0])
def test_prob_many_to_one_extremes(parent_fitness, p, rng):
    # For p=0, never random -> same as many_to_one
    # For p=1, always random -> indices can be anything, but still in range
    expected_exact = (np.arange(8) + 8 * np.array([3, 2, 1, 0, 0, 0, 3, 2])) if p == 0.0 else None
    result = prob_many_to_one(parent_fitness, OFFSPRING_FITNESS_LOCAL_SEARCH, rng, p)
    assert result.max() < 32
    assert result.min() >= 0
    assert len(result) == 8
    if p == 0.0:
        assert_array_equal(result, expected_exact)


@pytest.mark.parametrize(
    "parent_fitness, offspring_fitness",
    [
        (EXAMPLE_FITNESS, OFFSPRING_FITNESS_LOCAL_SEARCH),
    ],
)
@pytest.mark.parametrize("p", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_prob_many_to_one_shape(parent_fitness, offspring_fitness, p, rng):
    result = prob_many_to_one(parent_fitness, offspring_fitness, rng, p)
    assert result.max() < 32
    assert result.min() >= 0
    assert len(result) == 8


# ===================================================================
#  elitism
# ===================================================================
@pytest.mark.parametrize("population_fitness", [EXAMPLE_FITNESS])
@pytest.mark.parametrize(
    "offspring_fitness",
    [
        OFFSPRING_FITNESS_BETTER,
        OFFSPRING_FITNESS_WORSE,
        OFFSPRING_FITNESS_EQUAL,
        OFFSPRING_FITNESS_MIXED,
    ],
)
@pytest.mark.parametrize("amount", [0, 1, 5, 8, 10])
def test_elitism(population_fitness, offspring_fitness, amount):
    result = elitism(population_fitness, offspring_fitness, None, amount)
    assert result.max() < 16
    assert result.min() >= 0
    assert len(result) == 8
    assert np.all(result[:amount] < 8)
    assert np.all(result[amount:] >= 8)
    assert_array_equal(result[:amount], np.argsort(population_fitness)[::-1][:amount])


# ===================================================================
#  cond_elitism (same as elitism currently)
# ===================================================================
@pytest.mark.parametrize("population_fitness", [EXAMPLE_FITNESS])
@pytest.mark.parametrize(
    "offspring_fitness",
    [
        OFFSPRING_FITNESS_BETTER,
        OFFSPRING_FITNESS_WORSE,
        OFFSPRING_FITNESS_EQUAL,
        OFFSPRING_FITNESS_MIXED,
    ],
)
@pytest.mark.parametrize("amount", [0, 1, 5, 8, 10])
def test_cond_elitism(population_fitness, offspring_fitness, amount):
    result = cond_elitism(population_fitness, offspring_fitness, None, amount)
    assert result.max() < 16
    assert result.min() >= 0
    assert len(result) == 8
    assert np.all(result[:amount] < 8)
    assert np.all(result[amount:] >= 8)
    assert_array_equal(result[:amount], np.argsort(population_fitness)[::-1][:amount])


# ===================================================================
#  lamb_plus_mu
# ===================================================================
@pytest.mark.parametrize("population_fitness", [EXAMPLE_FITNESS])
@pytest.mark.parametrize(
    "offspring_fitness",
    [
        OFFSPRING_FITNESS_BETTER,
        OFFSPRING_FITNESS_WORSE,
        OFFSPRING_FITNESS_EQUAL,
        OFFSPRING_FITNESS_MIXED,
    ],
)
def test_lamb_plus_mu(population_fitness, offspring_fitness):
    result = keep_best(population_fitness, offspring_fitness, None)
    assert result.max() < 16
    assert result.min() >= 0
    assert len(result) == 8


# ===================================================================
#  lamb_comma_mu
# ===================================================================
@pytest.mark.parametrize("population_fitness", [EXAMPLE_FITNESS])
@pytest.mark.parametrize(
    "offspring_fitness",
    [
        OFFSPRING_FITNESS_BETTER,
        OFFSPRING_FITNESS_WORSE,
        OFFSPRING_FITNESS_EQUAL,
        OFFSPRING_FITNESS_MIXED,
    ],
)
def test_lamb_comma_mu(population_fitness, offspring_fitness):
    result = keep_best_offspring(population_fitness, offspring_fitness, None)
    assert result.max() < 16
    assert result.min() >= 8  # only offspring indices
    assert len(result) == 8
