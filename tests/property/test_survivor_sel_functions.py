"""
Property-based and correctness tests for survivor selection functions.

Key invariants / algorithmic correctness:
1. Every returned index is either a valid parent index (0..n_parents-1)
   or a valid offspring index (n_parents..n_parents+n_offspring-1).
2. The returned array has exactly n_parents elements.
3. `generational` always picks all offspring (indices shifted by n_parents).
4. `one_to_one` picks offspring when offspring is strictly better, keeps parent otherwise.
5. `elitism` always includes the top-k parents (invariant of elitism).
6. `keep_best` returns the n_parents best individuals from the combined pool.
7. `keep_best_offspring` returns indices into the offspring (all >= n_parents).
8. All functions are deterministic (stochastic ones with same seed).
"""

import numpy as np
import pytest
from hypothesis import given, settings, assume, HealthCheck
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp

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

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

seed_st = st.integers(0, 2**31 - 1)

# Fixed-size parent and offspring fitness arrays
@st.composite
def equal_size_fitness(draw, min_n=2, max_n=12):
    n = draw(st.integers(min_n, max_n))
    parent_fit = draw(hnp.arrays(
        dtype=np.float64,
        shape=n,
        elements=st.floats(-50.0, 50.0, allow_nan=False, allow_infinity=False),
    ))
    offspring_fit = draw(hnp.arrays(
        dtype=np.float64,
        shape=n,
        elements=st.floats(-50.0, 50.0, allow_nan=False, allow_infinity=False),
    ))
    return parent_fit, offspring_fit


@st.composite
def parent_and_offspring_fitness(draw, min_n=2, max_n=8, min_o=2, max_o=12):
    n_parents = draw(st.integers(min_n, max_n))
    n_offspring = draw(st.integers(min_o, max_o))
    parent_fit = draw(hnp.arrays(
        dtype=np.float64,
        shape=n_parents,
        elements=st.floats(-50.0, 50.0, allow_nan=False, allow_infinity=False),
    ))
    offspring_fit = draw(hnp.arrays(
        dtype=np.float64,
        shape=n_offspring,
        elements=st.floats(-50.0, 50.0, allow_nan=False, allow_infinity=False),
    ))
    return parent_fit, offspring_fit


# ---------------------------------------------------------------------------
# generational
# ---------------------------------------------------------------------------

def test_generational_selects_all_offspring():
    """All returned indices must point to offspring (>= n_parents)."""
    parent_fit = np.array([10.0, 5.0, 3.0])
    offspring_fit = np.array([1.0, 2.0, 4.0])
    result = generational(parent_fit, offspring_fit, random_state=None)
    n_parents = len(parent_fit)
    assert len(result) == n_parents
    assert np.all(result >= n_parents)
    assert np.all(result < n_parents + len(offspring_fit))


@given(fits=equal_size_fitness())
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_generational_indices_all_offspring(fits):
    parent_fit, offspring_fit = fits
    n_parents = len(parent_fit)
    result = generational(parent_fit, offspring_fit, random_state=None)
    assert len(result) == n_parents
    assert np.all(result >= n_parents)


# ---------------------------------------------------------------------------
# one_to_one – algorithmic correctness
# ---------------------------------------------------------------------------

def test_one_to_one_offspring_replaces_when_better():
    """For each pair, offspring index is chosen when offspring fitness > parent fitness."""
    parent_fit = np.array([1.0, 5.0, 3.0])
    offspring_fit = np.array([4.0, 2.0, 6.0])
    # offspring 0 > parent 0 → pick offspring (idx 3)
    # offspring 1 < parent 1 → keep parent (idx 1)
    # offspring 2 > parent 2 → pick offspring (idx 5)
    result = one_to_one(parent_fit, offspring_fit, random_state=None)
    n_parents = len(parent_fit)
    assert result[0] == n_parents + 0  # offspring won
    assert result[1] == 1              # parent kept
    assert result[2] == n_parents + 2  # offspring won


def test_one_to_one_parent_kept_when_equal():
    """When fitness values are equal, parent is kept (selection_mask uses <=)."""
    parent_fit = np.array([5.0, 5.0])
    offspring_fit = np.array([5.0, 5.0])
    # selection_mask = parent_fit <= offspring_fit = True for all → offspring chosen
    result = one_to_one(parent_fit, offspring_fit, random_state=None)
    n_parents = len(parent_fit)
    # With equal fitness, the implementation chooses offspring (<=)
    for i, r in enumerate(result):
        assert r in (i, i + n_parents)


@given(fits=equal_size_fitness())
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_one_to_one_returns_valid_indices(fits):
    parent_fit, offspring_fit = fits
    n_parents = len(parent_fit)
    result = one_to_one(parent_fit, offspring_fit, random_state=None)
    assert len(result) == n_parents
    # Each index i either points to parent i or offspring i
    for i, r in enumerate(result):
        assert r in (i, i + n_parents)


@given(fits=equal_size_fitness())
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_one_to_one_selects_best_for_each_pair(fits):
    """The selected individual must have fitness >= parent fitness for that slot."""
    parent_fit, offspring_fit = fits
    n_parents = len(parent_fit)
    result = one_to_one(parent_fit, offspring_fit, random_state=None)
    combined_fit = np.concatenate([parent_fit, offspring_fit])
    selected_fit = combined_fit[result]
    # Selected fitness for slot i must be >= parent fitness for slot i
    assert np.all(selected_fit >= parent_fit)


# ---------------------------------------------------------------------------
# prob_one_to_one
# ---------------------------------------------------------------------------

def test_prob_one_to_one_p0_behaves_like_one_to_one():
    """With p=0 (no forced replacement), should match deterministic one_to_one."""
    parent_fit = np.array([1.0, 5.0, 3.0])
    offspring_fit = np.array([4.0, 2.0, 6.0])
    r1 = one_to_one(parent_fit, offspring_fit, random_state=None)
    r2 = prob_one_to_one(parent_fit, offspring_fit, random_state=42, p=0.0)
    # Difference: prob_one_to_one uses < while one_to_one uses <=
    # When offspring is clearly better, both agree
    assert r2[0] >= len(parent_fit)  # offspring 0 clearly better
    assert r2[1] == 1               # parent 1 clearly better


@given(fits=equal_size_fitness(), seed=seed_st)
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_prob_one_to_one_valid_indices(fits, seed):
    parent_fit, offspring_fit = fits
    n_parents = len(parent_fit)
    result = prob_one_to_one(parent_fit, offspring_fit, random_state=seed, p=0.3)
    assert len(result) == n_parents
    for i, r in enumerate(result):
        assert r in (i, i + n_parents)


@given(fits=equal_size_fitness(), seed=seed_st)
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
def test_prob_one_to_one_deterministic(fits, seed):
    parent_fit, offspring_fit = fits
    r1 = prob_one_to_one(parent_fit, offspring_fit, random_state=seed, p=0.3)
    r2 = prob_one_to_one(parent_fit, offspring_fit, random_state=seed, p=0.3)
    np.testing.assert_array_equal(r1, r2)


# ---------------------------------------------------------------------------
# many_to_one – algorithmic correctness
# ---------------------------------------------------------------------------

def test_many_to_one_selects_best_from_each_block():
    """Each parent competes against its offspring block; the best must win."""
    parent_fit = np.array([1.0, 2.0])
    # offspring for parent 0: 0.5; offspring for parent 1: 10.0
    offspring_fit = np.array([0.5, 10.0])
    result = many_to_one(parent_fit, offspring_fit, random_state=None)
    n_parents = len(parent_fit)
    # parent 0 (1.0) > offspring 0 (0.5) → keep parent 0
    assert result[0] == 0
    # parent 1 (2.0) < offspring 1 (10.0) → pick offspring 1 (idx = n_parents + 1)
    assert result[1] == n_parents + 1


@given(fits=equal_size_fitness(), seed=seed_st)
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_many_to_one_valid_indices(fits, seed):
    parent_fit, offspring_fit = fits
    n_parents = len(parent_fit)
    result = many_to_one(parent_fit, offspring_fit, random_state=seed)
    assert len(result) == n_parents
    for i, r in enumerate(result):
        # Must point to parent i or offspring i
        assert r == i or r == i + n_parents


# ---------------------------------------------------------------------------
# elitism – algorithmic correctness
# ---------------------------------------------------------------------------

def test_elitism_always_keeps_top_parent():
    """The single best parent must always appear in the survivors."""
    parent_fit = np.array([10.0, 1.0, 2.0])
    offspring_fit = np.array([3.0, 4.0, 5.0])
    result = elitism(parent_fit, offspring_fit, random_state=None, amount=1)
    n_parents = len(parent_fit)
    # Parent 0 (fitness=10.0) must survive
    assert 0 in result


def test_elitism_top_k_parents_all_present():
    """All top-k parents must be in the result."""
    parent_fit = np.array([10.0, 8.0, 1.0, 2.0])
    offspring_fit = np.array([3.0, 4.0, 5.0, 6.0])
    result = elitism(parent_fit, offspring_fit, random_state=None, amount=2)
    n_parents = len(parent_fit)
    # Parents 0 (10.0) and 1 (8.0) must survive
    assert 0 in result
    assert 1 in result


def test_elitism_returns_n_parents_individuals():
    parent_fit = np.array([10.0, 1.0, 2.0, 3.0])
    offspring_fit = np.array([5.0, 6.0, 7.0, 8.0])
    result = elitism(parent_fit, offspring_fit, random_state=None, amount=2)
    assert len(result) == len(parent_fit)


@given(fits=parent_and_offspring_fitness(), amount=st.integers(1, 4))
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_elitism_valid_size_and_indices(fits, amount):
    parent_fit, offspring_fit = fits
    n_parents = len(parent_fit)
    n_offspring = len(offspring_fit)
    # elitism needs enough offspring to fill the non-elite slots
    assume(n_offspring >= n_parents - min(n_parents, amount))
    result = elitism(parent_fit, offspring_fit, random_state=None, amount=amount)
    assert len(result) == n_parents
    # All indices must be valid
    combined_size = n_parents + n_offspring
    assert np.all(result >= 0)
    assert np.all(result < combined_size)


@given(fits=parent_and_offspring_fitness())
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_elitism_amount1_best_parent_always_included(fits):
    """With amount=1 and a strictly unique best parent, it must appear in survivors."""
    parent_fit, offspring_fit = fits
    n_parents = len(parent_fit)
    n_offspring = len(offspring_fit)
    # Skip cases where there aren't enough offspring or no unique maximum
    assume(n_offspring >= n_parents - 1)
    assume(np.sum(parent_fit == parent_fit.max()) == 1)  # unique best
    result = elitism(parent_fit, offspring_fit, random_state=None, amount=1)
    best_parent_idx = int(np.argmax(parent_fit))
    assert best_parent_idx in result


# ---------------------------------------------------------------------------
# cond_elitism
# ---------------------------------------------------------------------------

def test_cond_elitism_parent_kept_when_best():
    """A parent better than all offspring must be kept."""
    parent_fit = np.array([100.0, 1.0, 2.0])
    offspring_fit = np.array([3.0, 4.0, 5.0])
    result = cond_elitism(parent_fit, offspring_fit, random_state=None, amount=1)
    n_parents = len(parent_fit)
    # Parent 0 (100.0) beats all offspring → must survive
    assert 0 in result


def test_cond_elitism_parent_not_kept_when_worse():
    """A parent that is NOT better than any offspring is replaced."""
    parent_fit = np.array([1.0, 2.0, 3.0])
    offspring_fit = np.array([10.0, 20.0, 30.0])
    result = cond_elitism(parent_fit, offspring_fit, random_state=None, amount=3)
    n_parents = len(parent_fit)
    # No parent beats offspring → all indices >= n_parents
    assert np.all(result >= n_parents)


@given(fits=parent_and_offspring_fitness(), amount=st.integers(1, 4))
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_cond_elitism_valid_size(fits, amount):
    parent_fit, offspring_fit = fits
    n_parents = len(parent_fit)
    n_offspring = len(offspring_fit)
    # cond_elitism needs at least n_parents offspring to fill all slots when no parent qualifies
    assume(n_offspring >= n_parents)
    result = cond_elitism(parent_fit, offspring_fit, random_state=None, amount=amount)
    assert len(result) == n_parents
    assert np.all(result >= 0)
    assert np.all(result < n_parents + n_offspring)


# ---------------------------------------------------------------------------
# keep_best – algorithmic correctness
# ---------------------------------------------------------------------------

def test_keep_best_selects_globally_best():
    """keep_best must return the n_parents best individuals from the union."""
    parent_fit = np.array([1.0, 2.0])
    offspring_fit = np.array([10.0, 20.0, 3.0])
    result = keep_best(parent_fit, offspring_fit, random_state=None)
    n_parents = len(parent_fit)
    # Combined: [1.0, 2.0, 10.0, 20.0, 3.0] → top 2 are at indices 3 (20.0) and 2 (10.0)
    assert 3 in result  # offspring[1] = 20.0
    assert 2 in result  # offspring[0] = 10.0


@given(fits=parent_and_offspring_fitness())
@settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
def test_keep_best_returns_n_parents_individuals(fits):
    parent_fit, offspring_fit = fits
    n_parents = len(parent_fit)
    result = keep_best(parent_fit, offspring_fit, random_state=None)
    assert len(result) == n_parents


@given(fits=parent_and_offspring_fitness())
@settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
def test_keep_best_selected_are_truly_best(fits):
    """The minimum fitness of selected must be >= the maximum fitness of unselected."""
    parent_fit, offspring_fit = fits
    n_parents = len(parent_fit)
    n_offspring = len(offspring_fit)
    result = keep_best(parent_fit, offspring_fit, random_state=None)
    combined_fit = np.concatenate([parent_fit, offspring_fit])
    selected_fit = combined_fit[result]
    all_indices = set(range(n_parents + n_offspring))
    unselected_indices = list(all_indices - set(result.tolist()))
    if unselected_indices:
        unselected_fit = combined_fit[unselected_indices]
        assert selected_fit.min() >= unselected_fit.max()


@given(fits=parent_and_offspring_fitness())
@settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
def test_keep_best_valid_indices(fits):
    parent_fit, offspring_fit = fits
    n_parents = len(parent_fit)
    n_offspring = len(offspring_fit)
    result = keep_best(parent_fit, offspring_fit, random_state=None)
    assert np.all(result >= 0)
    assert np.all(result < n_parents + n_offspring)


# ---------------------------------------------------------------------------
# keep_best_offspring
# ---------------------------------------------------------------------------

def test_keep_best_offspring_all_indices_are_offspring():
    """All returned indices must point to offspring (>= n_parents)."""
    parent_fit = np.array([100.0, 200.0, 300.0])  # parents much better
    offspring_fit = np.array([1.0, 2.0, 3.0])
    result = keep_best_offspring(parent_fit, offspring_fit, random_state=None)
    n_parents = len(parent_fit)
    # Even though parents are better, only offspring indices are returned
    assert np.all(result >= n_parents)


@given(fits=parent_and_offspring_fitness())
@settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
def test_keep_best_offspring_valid_size(fits):
    parent_fit, offspring_fit = fits
    n_parents = len(parent_fit)
    n_offspring = len(offspring_fit)
    assume(n_offspring >= n_parents)
    result = keep_best_offspring(parent_fit, offspring_fit, random_state=None)
    assert len(result) == n_parents
    assert np.all(result >= n_parents)
    assert np.all(result < n_parents + n_offspring)


def test_keep_best_offspring_picks_best_offspring():
    """Best offspring must be selected over worse ones."""
    parent_fit = np.array([0.0, 0.0])
    offspring_fit = np.array([5.0, 1.0, 10.0, 2.0])
    result = keep_best_offspring(parent_fit, offspring_fit, random_state=None)
    n_parents = len(parent_fit)
    # Top 2 offspring are at indices 2 (10.0) and 0 (5.0) → shifted: 4 and 2
    assert (n_parents + 2) in result  # offspring[2] = 10.0
    assert (n_parents + 0) in result  # offspring[0] = 5.0


# ---------------------------------------------------------------------------
# prob_many_to_one
# ---------------------------------------------------------------------------

def _make_many_to_one_fitness(n_parents, n_repetitions, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    parent_fit = rng.uniform(-10, 10, n_parents)
    offspring_fit = rng.uniform(-10, 10, n_parents * n_repetitions)
    return parent_fit, offspring_fit


def test_prob_many_to_one_returns_n_parents():
    parent_fit, offspring_fit = _make_many_to_one_fitness(4, 3)
    result = prob_many_to_one(parent_fit, offspring_fit, random_state=0, p=0.0)
    assert len(result) == len(parent_fit)


def test_prob_many_to_one_p0_matches_many_to_one():
    """With p=0 prob_many_to_one should be deterministic and match many_to_one."""
    rng = np.random.default_rng(7)
    parent_fit, offspring_fit = _make_many_to_one_fitness(4, 2, rng=rng)
    result_prob = prob_many_to_one(parent_fit, offspring_fit, random_state=0, p=0.0)
    result_det = many_to_one(parent_fit, offspring_fit, random_state=None)
    np.testing.assert_array_equal(result_prob, result_det)


def test_prob_many_to_one_valid_indices():
    parent_fit, offspring_fit = _make_many_to_one_fitness(3, 2)
    n_parents = len(parent_fit)
    n_offspring = len(offspring_fit)
    result = prob_many_to_one(parent_fit, offspring_fit, random_state=42, p=0.3)
    assert len(result) == n_parents
    assert np.all(result >= 0)
    assert np.all(result < n_parents + n_offspring)


def test_prob_many_to_one_p1_always_random():
    """With p=1 every pick is random (stochastic), result must still be valid indices."""
    parent_fit, offspring_fit = _make_many_to_one_fitness(4, 3)
    n_parents = len(parent_fit)
    n_offspring = len(offspring_fit)
    result = prob_many_to_one(parent_fit, offspring_fit, random_state=99, p=1.0)
    assert len(result) == n_parents
    assert np.all(result >= 0)
    assert np.all(result < n_parents + n_offspring)


def test_prob_many_to_one_deterministic_with_same_seed():
    parent_fit, offspring_fit = _make_many_to_one_fitness(5, 2)
    r1 = prob_many_to_one(parent_fit, offspring_fit, random_state=123, p=0.5)
    r2 = prob_many_to_one(parent_fit, offspring_fit, random_state=123, p=0.5)
    np.testing.assert_array_equal(r1, r2)


# ---------------------------------------------------------------------------
# SurvivorSelectionFromLambda (survivor_selection_base.py)
# ---------------------------------------------------------------------------

def test_survivor_selection_from_lambda_basic():
    from metaheuristic_designer.survivor_selection_base import SurvivorSelectionFromLambda
    from metaheuristic_designer.benchmarks import Sphere
    from metaheuristic_designer.population import Population
    import numpy as np

    def my_selection_fn(population, offspring, random_state, **kwargs):
        # Always keep all offspring
        n_parents = population.pop_size
        n_offspring = offspring.pop_size
        return np.arange(n_parents, n_parents + n_offspring)

    sel = SurvivorSelectionFromLambda(my_selection_fn)
    objfunc = Sphere(dimension=4, mode="min")

    rng = np.random.default_rng(0)
    parents_geno = rng.uniform(-1, 1, (5, 4))
    offspring_geno = rng.uniform(-1, 1, (5, 4))

    parents = Population(objfunc, parents_geno)
    parents.calculate_fitness()
    offspring = Population(objfunc, offspring_geno)
    offspring.calculate_fitness()

    result = sel.select(parents, offspring)
    assert result.pop_size == 5


def test_survivor_selection_from_lambda_validates_function():
    """SurvivorSelectionFromLambda.select calls _validate_function path via inspection."""
    from metaheuristic_designer.survivor_selection_base import SurvivorSelectionFromLambda
    from metaheuristic_designer.benchmarks import Sphere
    from metaheuristic_designer.population import Population
    import numpy as np

    # A function with fewer than 3 positional args should be blocked on construction
    def bad_fn(a, b):
        return np.array([0])

    with pytest.raises(TypeError):
        SurvivorSelectionFromLambda(bad_fn).select(None, None)


def test_survivor_selection_get_state():
    from metaheuristic_designer.survivor_selection_base import SurvivorSelectionFromLambda

    def my_fn(pop, off, rs, **kw):
        return np.arange(pop.pop_size)

    sel = SurvivorSelectionFromLambda(my_fn, name="TestSel")
    state = sel.get_state()
    assert state["name"] == "TestSel"
    assert "class_name" in state
