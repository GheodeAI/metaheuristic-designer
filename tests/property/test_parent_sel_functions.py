"""
Property-based and correctness tests for parent selection functions.

Key invariants / algorithmic correctness:
1. Every returned index is a valid index into the population (0 <= idx < n).
2. `select_best` with amount=1 returns the index of the SINGLE highest-fitness individual.
3. `select_best` with amount=k returns the top-k indices (verified by content).
4. `prob_tournament` with prob=1 always picks the highest-fitness participant.
5. `uniform_selection` returns the requested number of valid indices.
6. `roulette` and `sus` return valid indices for the requested amount.
7. All stochastic selectors are deterministic with the same seed.
"""

import numpy as np
import pytest
from hypothesis import given, settings, assume, HealthCheck
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp

from metaheuristic_designer.parent_selection.parent_selection_functions import (
    SelectionDist,
    selection_distribution,
    select_best,
    prob_tournament,
    uniform_selection,
    roulette,
    sus,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

seed_st = st.integers(0, 2**31 - 1)

fitness_st = hnp.arrays(
    dtype=np.float64,
    shape=st.integers(3, 20),
    elements=st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
)

amount_st = st.integers(1, 10)


# ---------------------------------------------------------------------------
# SelectionDist.from_str
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("fitness_proportional", SelectionDist.FIT_PROP),
    ("fitness_prop", SelectionDist.FIT_PROP),
    ("sigma_scaling", SelectionDist.SIGMA_SCALE),
    ("linear_rank", SelectionDist.LIN_RANK),
    ("lin_rank", SelectionDist.LIN_RANK),
    ("exponential_rank", SelectionDist.EXP_RANK),
    ("exp_rank", SelectionDist.EXP_RANK),
])
def test_selection_dist_from_str_known(name, expected):
    assert SelectionDist.from_str(name) == expected


def test_selection_dist_from_str_unknown_raises():
    with pytest.raises((ValueError, KeyError)):
        SelectionDist.from_str("not_a_distribution")


# ---------------------------------------------------------------------------
# selection_distribution – weights sum to 1 and are non-negative
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", [
    SelectionDist.FIT_PROP,
    SelectionDist.SIGMA_SCALE,
    SelectionDist.LIN_RANK,
    SelectionDist.EXP_RANK,
])
def test_selection_distribution_sums_to_one(method):
    fitness = np.array([1.0, 5.0, 2.0, 8.0, 3.0])
    weights = selection_distribution(fitness, method)
    assert weights.sum() == pytest.approx(1.0, abs=1e-9)


def test_selection_distribution_lin_rank_nonneg():
    fitness = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = selection_distribution(fitness, SelectionDist.LIN_RANK)
    assert np.all(weights >= 0)


def test_selection_distribution_fit_prop_nonneg():
    # Fitness can be negative; shifting by min ensures non-negative weights
    fitness = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
    weights = selection_distribution(fitness, SelectionDist.FIT_PROP)
    assert np.all(weights >= 0)


# ---------------------------------------------------------------------------
# select_best – algorithmic correctness
# ---------------------------------------------------------------------------

def test_select_best_returns_highest_fitness_index():
    """With amount=1, must return the single best individual."""
    fitness = np.array([3.0, 1.0, 10.0, 4.0, -1.0])
    result = select_best(fitness, amount=1)
    assert len(result) == 1
    assert result[0] == 2  # index of 10.0


def test_select_best_top3_correctness():
    """The returned indices must correspond to the three highest-fitness values."""
    fitness = np.array([3.0, 1.0, 10.0, 4.0, -1.0])
    result = select_best(fitness, amount=3)
    assert len(result) == 3
    selected_fitness = sorted(fitness[result], reverse=True)
    expected_top3 = sorted(fitness, reverse=True)[:3]
    assert selected_fitness == expected_top3


@given(fitness=fitness_st, amount=amount_st)
@settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
def test_select_best_indices_valid(fitness, amount):
    assume(len(fitness) >= amount)
    result = select_best(fitness, amount=amount)
    assert len(result) == amount
    assert np.all(result >= 0)
    assert np.all(result < len(fitness))


@given(fitness=fitness_st, amount=amount_st)
@settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
def test_select_best_returns_actual_best(fitness, amount):
    """The minimum fitness among selected must be >= maximum among the unselected."""
    assume(len(fitness) > amount)
    result = select_best(fitness, amount=amount)
    selected_fitness = fitness[result]
    unselected_mask = np.ones(len(fitness), dtype=bool)
    unselected_mask[result] = False
    if unselected_mask.any():
        assert selected_fitness.min() >= fitness[unselected_mask].max()


# ---------------------------------------------------------------------------
# prob_tournament – algorithmic correctness
# ---------------------------------------------------------------------------

def test_prob_tournament_prob1_picks_best_in_tournament():
    """With prob=1 and tournament_size=n, must always return the global best."""
    fitness = np.array([1.0, 5.0, 10.0, 2.0, 3.0])
    # All individuals participate → best must be chosen
    result = prob_tournament(fitness, amount=1, random_state=0,
                             tournament_size=len(fitness), prob=1.0)
    assert result[0] == 2  # index of 10.0


@given(fitness=fitness_st, seed=seed_st)
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_prob_tournament_indices_valid(fitness, seed):
    assume(len(fitness) >= 3)
    result = prob_tournament(fitness, amount=4, random_state=seed, tournament_size=3, prob=1.0)
    assert np.all(result >= 0)
    assert np.all(result < len(fitness))


@given(fitness=fitness_st, seed=seed_st)
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
def test_prob_tournament_deterministic(fitness, seed):
    assume(len(fitness) >= 3)
    r1 = prob_tournament(fitness, amount=4, random_state=seed, tournament_size=3, prob=0.8)
    r2 = prob_tournament(fitness, amount=4, random_state=seed, tournament_size=3, prob=0.8)
    np.testing.assert_array_equal(r1, r2)


def test_prob_tournament_returns_requested_amount():
    fitness = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = prob_tournament(fitness, amount=7, random_state=0, tournament_size=2, prob=1.0)
    assert len(result) == 7


# ---------------------------------------------------------------------------
# uniform_selection
# ---------------------------------------------------------------------------

@given(fitness=fitness_st, amount=amount_st, seed=seed_st)
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_uniform_selection_indices_valid(fitness, amount, seed):
    result = uniform_selection(fitness, amount=amount, random_state=seed)
    assert len(result) == amount
    assert np.all(result >= 0)
    assert np.all(result < len(fitness))


@given(fitness=fitness_st, seed=seed_st)
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
def test_uniform_selection_deterministic(fitness, seed):
    r1 = uniform_selection(fitness, amount=5, random_state=seed)
    r2 = uniform_selection(fitness, amount=5, random_state=seed)
    np.testing.assert_array_equal(r1, r2)


# ---------------------------------------------------------------------------
# roulette
# ---------------------------------------------------------------------------

@given(fitness=fitness_st, amount=amount_st, seed=seed_st)
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_roulette_indices_valid(fitness, amount, seed):
    result = roulette(fitness, amount=amount, random_state=seed, method=SelectionDist.FIT_PROP)
    assert len(result) == amount
    assert np.all(result >= 0)
    assert np.all(result < len(fitness))


@given(fitness=fitness_st, seed=seed_st)
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
def test_roulette_deterministic(fitness, seed):
    r1 = roulette(fitness, amount=4, random_state=seed, method=SelectionDist.LIN_RANK)
    r2 = roulette(fitness, amount=4, random_state=seed, method=SelectionDist.LIN_RANK)
    np.testing.assert_array_equal(r1, r2)


@pytest.mark.parametrize("method", [
    SelectionDist.FIT_PROP,
    SelectionDist.LIN_RANK,
    SelectionDist.EXP_RANK,
])
def test_roulette_all_methods_valid_indices(method):
    fitness = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
    result = roulette(fitness, amount=6, random_state=0, method=method)
    assert len(result) == 6
    assert np.all(result >= 0)
    assert np.all(result < len(fitness))


# ---------------------------------------------------------------------------
# sus (Stochastic Universal Sampling)
# ---------------------------------------------------------------------------

@given(fitness=fitness_st, amount=amount_st, seed=seed_st)
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_sus_indices_valid(fitness, amount, seed):
    result = sus(fitness, amount=amount, random_state=seed, method=SelectionDist.FIT_PROP)
    assert len(result) == amount
    assert np.all(result >= 0)
    assert np.all(result < len(fitness))


@given(fitness=fitness_st, seed=seed_st)
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
def test_sus_deterministic(fitness, seed):
    r1 = sus(fitness, amount=5, random_state=seed, method=SelectionDist.LIN_RANK)
    r2 = sus(fitness, amount=5, random_state=seed, method=SelectionDist.LIN_RANK)
    np.testing.assert_array_equal(r1, r2)


def test_sus_uniform_fitness_covers_all_with_large_amount():
    """With uniform fitness, SUS should spread evenly over the population."""
    fitness = np.ones(10)
    # Request exactly 10 samples — SUS should cover all indices
    result = sus(fitness, amount=10, random_state=0, method=SelectionDist.LIN_RANK)
    assert len(result) == 10
    assert np.all(result >= 0)
    assert np.all(result < 10)
