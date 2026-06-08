# tests/test_parent_selection_factory.py
import pytest
import numpy as np
from numpy.testing import assert_array_equal

from conftest import dummy_objfunc, rng
from conftest import make_pop  # plain function from conftest

from metaheuristic_designer.parent_selection import (
    ParentSelectionDef,
    ParentSelectionFromLambda,
    NullParentSelection,
    create_parent_selection,
)
from metaheuristic_designer.population import Population


# ===================================================================
#  ParentSelectionDef – direct call
# ===================================================================
def test_parent_selection_def_calls_wrapped_function():
    def dummy(fitness, amount, rng):
        return np.array([0, 2])

    def_obj = ParentSelectionDef(dummy)
    pop = make_pop([1.0, 2.0], dummy_objfunc)
    result = def_obj(pop, amount=2, rng=rng)
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
    def_obj(pop, amount=1, rng=rng)

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


# ===================================================================
#  Integration: select() returns a valid subset of the population
# ===================================================================
@pytest.mark.parametrize(
    "method, kwargs",
    [
        ("best", {}),
        ("tournament", {"tournament_size": 2}),
        ("random", {}),
        ("roulette", {}),
        ("sus", {}),
    ],
)
def test_factory_select_returns_valid_parents(method, kwargs, rng, dummy_objfunc):
    population = make_pop([5.0, 1.0, 3.0, 2.0], dummy_objfunc)
    amount = 3

    sel = create_parent_selection(method, rng=rng, **kwargs)
    parents = sel.select(population, amount)

    # Must be a Population with the requested amount
    assert len(parents) == amount

    # All parents must be rows from the original population
    for row in parents.genotype_matrix:
        assert any(np.array_equal(row, population.genotype_matrix[i]) for i in range(len(population)))

    # For "best", the first parent must be the individual with highest fitness
    if method == "best":
        best_idx = np.argmax(population.fitness)
        assert np.array_equal(parents.genotype_matrix[0], population.genotype_matrix[best_idx])


# ===================================================================
#  Integration: select() returns a valid subset of the population
# ===================================================================
@pytest.mark.parametrize(
    "method, kwargs",
    [
        ("best", {}),
        ("tournament", {"tournament_size": 2}),
        ("random", {}),
        ("roulette", {}),
        ("sus", {}),
    ],
)
def test_factory_select_returns_valid_parents(method, kwargs, rng, dummy_objfunc):
    population = make_pop([5.0, 1.0, 3.0, 2.0], dummy_objfunc)
    amount = 3

    sel = create_parent_selection(method, rng=rng, **kwargs)
    parents = sel.select(population, amount)

    assert len(parents) == amount

    # All parents must be rows from the original population
    for row in parents.genotype_matrix:
        assert any(np.array_equal(row, population.genotype_matrix[i]) for i in range(len(population)))

    if method == "best":
        best_idx = np.argmax(population.fitness)
        assert np.array_equal(parents.genotype_matrix[0], population.genotype_matrix[best_idx])


def test_null_parent_selection_returns_original(rng, dummy_objfunc):
    population = make_pop([1.0, 2.0], dummy_objfunc)
    sel = NullParentSelection()
    result = sel.select(population, None)
    assert len(result) == len(population)
    assert_array_equal(result.genotype_matrix, population.genotype_matrix)
