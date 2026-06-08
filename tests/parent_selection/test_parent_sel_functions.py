# tests/test_parent_selection_functions.py
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

# Conftest constants
from conftest import EXAMPLE_FITNESS, make_pop, dummy_objfunc, rng

# Functions under test – updated imports after refactoring
from metaheuristic_designer.parent_selection import (
    select_best,
    prob_tournament,
    uniform_selection,
    shuffle_population,
    roulette,
    sus,
    create_scaling_fn,
    create_parent_selection,
    ParentSelectionDef,
    ParentSelectionFromLambda,
    NullParentSelection,
)


# ===================================================================
#  select_best
# ===================================================================
@pytest.mark.parametrize(
    "fitness, amount, expected",
    [
        (EXAMPLE_FITNESS, 3, np.array([7, 6, 5])),
        (EXAMPLE_FITNESS, 8, np.array([7, 6, 5, 4, 3, 2, 1, 0])),
        (EXAMPLE_FITNESS, 0, np.array([], dtype=int)),
        (np.array([5.0, -3.0, 2.0]), 2, np.array([0, 2])),
    ],
)
def test_select_best(fitness, amount, expected):
    result = select_best(fitness, amount)
    assert_array_equal(result, expected)


# ===================================================================
#  uniform_selection (needs `rng` fixture)
# ===================================================================
@pytest.mark.parametrize("amount", [0, 3, 5])
def test_uniform_selection(amount, rng):
    result = uniform_selection(EXAMPLE_FITNESS, amount, rng=rng)
    assert len(result) == amount
    if amount > 0:
        assert result.max() < len(EXAMPLE_FITNESS)
        assert result.min() >= 0


def test_uniform_selection_reproducible(rng):
    amount = 4
    expected = uniform_selection(EXAMPLE_FITNESS, amount, rng=np.random.default_rng(42))
    result = uniform_selection(EXAMPLE_FITNESS, amount, rng=rng)
    assert_array_equal(result, expected)


# ===================================================================
#  shuffle_population (new function)
# ===================================================================
def test_shuffle_small_amount(rng):
    amount = 3
    pop_size = len(EXAMPLE_FITNESS)
    result = shuffle_population(EXAMPLE_FITNESS, amount, rng=rng)
    assert len(result) == amount
    # all indices must be distinct and within range
    assert len(set(result)) == amount
    assert result.max() < pop_size


def test_shuffle_large_amount(rng):
    amount = 12  # > population size 8
    pop_size = len(EXAMPLE_FITNESS)
    result = shuffle_population(EXAMPLE_FITNESS, amount, rng=rng)
    assert len(result) == amount
    # every index must appear at least once
    assert set(range(pop_size)).issubset(set(result))


# ===================================================================
#  prob_tournament (needs `rng` fixture)
# ===================================================================
def test_prob_tournament_prob_1(rng):
    amount = 4
    expected = prob_tournament(EXAMPLE_FITNESS, amount, rng=np.random.default_rng(42), tournament_size=3, prob=1.0)
    result = prob_tournament(EXAMPLE_FITNESS, amount, rng=rng, tournament_size=3, prob=1.0)
    assert_array_equal(result, expected)


def test_prob_tournament_prob_0(rng):
    amount = 4
    expected = prob_tournament(EXAMPLE_FITNESS, amount, rng=np.random.default_rng(42), tournament_size=3, prob=0.0)
    result = prob_tournament(EXAMPLE_FITNESS, amount, rng=rng, tournament_size=3, prob=0.0)
    assert_array_equal(result, expected)


@pytest.mark.parametrize("amount", [0, 3, 8])
@pytest.mark.parametrize("prob", [0.1, 0.5, 0.9])
def test_prob_tournament_shape(amount, prob, rng):
    result = prob_tournament(EXAMPLE_FITNESS, amount, rng=rng, tournament_size=3, prob=prob)
    assert len(result) == amount
    if amount > 0:
        assert result.max() < len(EXAMPLE_FITNESS)
        assert result.min() >= 0


# ===================================================================
#  create_scaling_fn (replaces old selection_distribution tests)
# ===================================================================
def test_scaling_fitness_proportional():
    fitness = np.array([1, 2, 3], dtype=float)
    fn = create_scaling_fn("fitness_proportional", scaling_factor=1)
    weights = fn(fitness)
    expected = np.array([1 / 6, 1 / 3, 1 / 2])
    assert_array_equal(weights, expected)


def test_scaling_sigma_scaling():
    fitness = np.array([1.0, 2.0, 3.0])
    fn = create_scaling_fn("sigma_scaling", scaling_factor=1)
    # sigma_scaling sets weights = max(0, fitness - (mean - std))
    # mean=2, std=0.816... => mean - std ≈ 1.184
    # weights = max(0, [1-1.184,2-1.184,3-1.184]) = max(0, [-0.184,0.816,1.816]) = [0,0.816,1.816]
    # normalized sum = 2.632 => [0,0.310,0.690]
    weights = fn(fitness)
    assert np.all(weights >= 0)
    assert np.isclose(weights.sum(), 1.0)


def test_scaling_linear_rank():
    fitness = np.array([10.0, 20.0, 30.0, 40.0])
    fn = create_scaling_fn("linear_ranking", scaling_factor=0)  # f=0 -> extreme non‑linear
    weights = fn(fitness)
    # ranks: worst=0, best=3
    # linear_ranking with f=0: (2-0)+(2*rank*(0-1))/(3) => 2 - (2*rank/3)
    # ranks: 0,1,2,3 -> weights: 2, 4/3, 2/3, 0 -> normalized sum = 4 => [0.5, 0.333, 0.167, 0]
    expected = np.array([0.5, 1 / 3, 1 / 6, 0.0])
    assert_allclose(weights, expected, atol=1e-6)


def test_scaling_exponential_rank():
    fitness = np.array([1.0, 2.0, 3.0])
    fn = create_scaling_fn("exponential_ranking", scaling_factor=None)
    weights = fn(fitness)
    # ranks: 0,1,2  => 1-exp(-0), 1-exp(-1), 1-exp(-2) ≈ [0,0.632,0.865]
    # normalised roughly [0,0.422,0.578]
    assert np.isclose(weights.sum(), 1.0)
    assert weights[0] < weights[1] < weights[2]


# ===================================================================
#  roulette (needs `rng` fixture)  – updated to string methods
# ===================================================================
@pytest.mark.parametrize(
    "method, scaling_factor",
    [
        ("fitness_proportional", 1),
        ("sigma_scaling", 1),
        ("linear_ranking", 0),  # extreme linear ranking
        ("exponential_ranking", None),
    ],
)
def test_roulette_deterministic(method, scaling_factor, rng):
    amount = 5
    expected = roulette(EXAMPLE_FITNESS, amount, rng=np.random.default_rng(42), method=method, scaling_factor=scaling_factor)
    result = roulette(EXAMPLE_FITNESS, amount, rng=rng, method=method, scaling_factor=scaling_factor)
    assert_array_equal(result, expected)


def test_roulette_sanity(rng):
    amount = 4
    result = roulette(EXAMPLE_FITNESS, amount, rng=rng)
    assert len(result) == amount
    assert result.max() < len(EXAMPLE_FITNESS)
    assert result.min() >= 0


# ===================================================================
#  sus (needs `rng` fixture)  – updated to string methods
# ===================================================================
@pytest.mark.parametrize(
    "method, scaling_factor",
    [
        ("fitness_proportional", 1),
        ("sigma_scaling", 1),
        ("linear_ranking", 0),
        ("exponential_ranking", None),
    ],
)
def test_sus_deterministic(method, scaling_factor, rng):
    amount = 5
    expected = sus(EXAMPLE_FITNESS, amount, rng=np.random.default_rng(42), method=method, scaling_factor=scaling_factor)
    result = sus(EXAMPLE_FITNESS, amount, rng=rng, method=method, scaling_factor=scaling_factor)
    assert_array_equal(result, expected)


def test_sus_sanity(rng):
    amount = 4
    result = sus(EXAMPLE_FITNESS, amount, rng=rng)
    assert len(result) == amount
    assert result.max() < len(EXAMPLE_FITNESS)
    assert result.min() >= 0


# ===================================================================
#  ParentSelectionDef – direct call
# ===================================================================
def test_parent_selection_def_calls_wrapped_function():
    def dummy(fitness, amount, rng):
        return np.array([0, 2])

    def_obj = ParentSelectionDef(dummy)
    pop = make_pop([1.0, 2.0], dummy_objfunc)
    result = def_obj(pop, amount=2, rng=np.random.default_rng())
    assert_array_equal(result, [0, 2])


def test_parent_selection_def_passes_fitness_and_kwargs():
    captured = {}

    def spy(fitness, amount, rng, **kw):
        captured["fitness"] = fitness
        captured["amount"] = amount
        captured["kw"] = kw
        return np.array([0])

    def_obj = ParentSelectionDef(spy, params={"extra": 5})
    pop = make_pop([10.0, 20.0], dummy_objfunc)
    def_obj(pop, amount=1, rng=np.random.default_rng())

    assert_array_equal(captured["fitness"], [10.0, 20.0])
    assert captured["amount"] == 1
    assert captured["kw"]["extra"] == 5


# ===================================================================
#  create_parent_selection – type and name
# ===================================================================
@pytest.mark.parametrize(
    "method, expected_type",
    [
        ("best", ParentSelectionFromLambda),
        ("tournament", ParentSelectionFromLambda),
        ("random", ParentSelectionFromLambda),
        ("shuffle", ParentSelectionFromLambda),  # new entry
        ("roulette", ParentSelectionFromLambda),
        ("sus", ParentSelectionFromLambda),
        ("nothing", NullParentSelection),
    ],
)
def test_create_returns_correct_type(method, expected_type, rng):
    sel = create_parent_selection(method, rng=rng)
    assert isinstance(sel, expected_type)


def test_create_uses_given_name(rng):
    sel = create_parent_selection("tournament", name="custom_name", rng=rng)
    assert sel.name == "custom_name"


def test_create_default_name_is_method(rng):
    sel = create_parent_selection("truncation", rng=rng)
    assert sel.name == "truncation"
