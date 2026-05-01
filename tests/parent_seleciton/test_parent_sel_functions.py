# tests/test_parent_selection_functions.py
import pytest
import numpy as np
from numpy.testing import assert_array_equal

# Conftest constants
from conftest import EXAMPLE_FITNESS, make_pop, dummy_objfunc, rng

# Functions under test
from metaheuristic_designer.parent_selection import (
    select_best,
    prob_tournament,
    uniform_selection,
    roulette,
    sus,
    selection_distribution,
    create_parent_selection,
    SelectionDist,
    ParentSelectionDef,
    ParentSelectionFromLambda,
    NullParentSelection
)


# ===================================================================
#  select_best
# ===================================================================
@pytest.mark.parametrize("fitness, amount, expected", [
    (EXAMPLE_FITNESS, 3, np.array([7, 6, 5])),
    (EXAMPLE_FITNESS, 8, np.array([7, 6, 5, 4, 3, 2, 1, 0])),
    (EXAMPLE_FITNESS, 0, np.array([])),
    (np.array([5.0, -3.0, 2.0]), 2, np.array([0, 2])),
])
def test_select_best(fitness, amount, expected):
    result = select_best(fitness, amount)
    assert_array_equal(result, expected)


# ===================================================================
#  uniform_selection (needs `rng` fixture)
# ===================================================================
@pytest.mark.parametrize("fitness", [EXAMPLE_FITNESS])
@pytest.mark.parametrize("amount, expected", [
    (0, np.array([], dtype=int)),
    (3, np.random.default_rng(42).integers(0, len(EXAMPLE_FITNESS), 3)),
    (5, np.random.default_rng(42).integers(0, len(EXAMPLE_FITNESS), 5)),
])
def test_uniform_selection(fitness, amount, expected, rng):
    result = uniform_selection(fitness, amount, random_state=rng)
    assert_array_equal(result, expected)
    if amount > 0:
        assert result.dtype == np.intp
        assert result.max() < len(fitness)
        assert result.min() >= 0


# ===================================================================
#  prob_tournament (needs `rng` fixture)
# ===================================================================
def test_prob_tournament_prob_1(rng):
    amount = 4
    expected = prob_tournament(EXAMPLE_FITNESS, amount, random_state=np.random.default_rng(42), tournament_size=3, prob=1.0)
    result = prob_tournament(EXAMPLE_FITNESS, amount, random_state=rng, tournament_size=3, prob=1.0)
    assert_array_equal(result, expected)

def test_prob_tournament_prob_0(rng):
    amount = 4
    expected = prob_tournament(EXAMPLE_FITNESS, amount, random_state=np.random.default_rng(42), tournament_size=3, prob=0.0)
    result = prob_tournament(EXAMPLE_FITNESS, amount, random_state=rng, tournament_size=3, prob=0.0)
    assert_array_equal(result, expected)

@pytest.mark.parametrize("fitness", [EXAMPLE_FITNESS])
@pytest.mark.parametrize("amount", [0, 3, 8])
@pytest.mark.parametrize("prob", [0.1, 0.5, 0.9])
def test_prob_tournament_shape(fitness, amount, prob, rng):
    result = prob_tournament(fitness, amount, random_state=rng, tournament_size=3, prob=prob)
    assert len(result) == amount
    if amount > 0:
        assert result.max() < len(fitness)
        assert result.min() >= 0


# ===================================================================
#  roulette (needs `rng` fixture)
# ===================================================================
@pytest.mark.parametrize("fitness, method, f", [
    (EXAMPLE_FITNESS, SelectionDist.FIT_PROP, None),
    (EXAMPLE_FITNESS, SelectionDist.SIGMA_SCALE, None),
    (EXAMPLE_FITNESS, SelectionDist.LIN_RANK, None),
    (EXAMPLE_FITNESS, SelectionDist.EXP_RANK, None),
])
def test_roulette_deterministic(fitness, method, f, rng):
    amount = 5
    expected = roulette(fitness, amount, random_state=np.random.default_rng(42), method=method, f=f)
    result = roulette(fitness, amount, random_state=rng, method=method, f=f)
    assert_array_equal(result, expected)

def test_roulette_sanity(rng):
    amount = 4
    result = roulette(EXAMPLE_FITNESS, amount, random_state=rng)
    assert len(result) == amount
    assert result.max() < len(EXAMPLE_FITNESS)
    assert result.min() >= 0


# ===================================================================
#  sus (needs `rng` fixture)
# ===================================================================
@pytest.mark.parametrize("fitness, method, f", [
    (EXAMPLE_FITNESS, SelectionDist.FIT_PROP, None),
    (EXAMPLE_FITNESS, SelectionDist.SIGMA_SCALE, None),
    (EXAMPLE_FITNESS, SelectionDist.LIN_RANK, None),
    (EXAMPLE_FITNESS, SelectionDist.EXP_RANK, None),
])
def test_sus_deterministic(fitness, method, f, rng):
    amount = 5
    expected = sus(fitness, amount, random_state=np.random.default_rng(42), method=method, f=f)
    result = sus(fitness, amount, random_state=rng, method=method, f=f)
    assert_array_equal(result, expected)

def test_sus_sanity(rng):
    amount = 4
    result = sus(EXAMPLE_FITNESS, amount, random_state=rng)
    assert len(result) == amount
    assert result.max() < len(EXAMPLE_FITNESS)
    assert result.min() >= 0


# ===================================================================
#  selection_distribution (deterministic, no rng needed)
# ===================================================================
def test_selection_distribution_fit_prop():
    fitness = np.array([1.0, 2.0, 3.0])
    weights = selection_distribution(fitness, SelectionDist.FIT_PROP, f=1.0)
    expected = np.array([1/6, 1/3, 1/2])
    assert_array_equal(weights, expected)

def test_selection_distribution_negative_fitness():
    fitness = np.array([-5.0, 0.0, 5.0])
    weights = selection_distribution(fitness, SelectionDist.FIT_PROP, f=1.0)
    expected = np.array([1/18, 6/18, 11/18])
    assert_array_equal(weights, expected)

def test_selection_distribution_linear_rank():
    fitness = np.array([10.0, 20.0, 30.0, 40.0])
    weights = selection_distribution(fitness, SelectionDist.LIN_RANK)
    expected = np.array([0.0, 1/6, 1/3, 1/2])
    assert_array_equal(weights, expected)

def test_selection_distribution_linear_rank_f_greater_than_2():
    fitness = np.array([1.0, 2.0, 3.0])
    weights = selection_distribution(fitness, SelectionDist.LIN_RANK, f=5.0)
    expected = np.array([0.0, 1/3, 2/3])
    assert_array_equal(weights, expected)

# ===================================================================
#  ParentSelectionDef – direct call
# ===================================================================
def test_parent_selection_def_calls_wrapped_function():
    def dummy(fitness, amount, rng):
        return np.array([0, 2])

    def_obj = ParentSelectionDef(dummy)
    pop = make_pop([1.0, 2.0], dummy_objfunc)
    result = def_obj(pop, amount=2, random_state=rng)
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
    def_obj(pop, amount=1, random_state=rng)

    assert_array_equal(captured["fitness"], [10.0, 20.0])
    assert captured["amount"] == 1
    assert captured["kw"]["extra"] == 5


# ===================================================================
#  create_parent_selection – type and name
# ===================================================================
@pytest.mark.parametrize("method, expected_type", [
    ("best", ParentSelectionFromLambda),
    ("tournament", ParentSelectionFromLambda),
    ("random", ParentSelectionFromLambda),
    ("roulette", ParentSelectionFromLambda),
    ("sus", ParentSelectionFromLambda),
    ("nothing", NullParentSelection),
])
def test_create_returns_correct_type(method, expected_type, rng):
    sel = create_parent_selection(method, random_state=rng)
    assert isinstance(sel, expected_type)


def test_create_uses_given_name(rng):
    sel = create_parent_selection("tournament", name="custom_name", random_state=rng)
    assert sel.name == "custom_name"


def test_create_default_name_is_method(rng):
    sel = create_parent_selection("truncation", random_state=rng)
    assert sel.name == "truncation"