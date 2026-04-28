import pytest
import numpy as np
from numpy.testing import assert_array_equal

# Conftest fixtures and helpers
from conftest import dummy_objfunc, rng
from conftest import make_pop   # plain function, NOT a fixture

# Factory and related classes
from metaheuristic_designer.survivor_selection_methods import (
    SurvivorSelectionDef,
    SurvivorSelectionFromLambda,
    NullSurvivorSelection,
    create_survivor_selection,
)

from metaheuristic_designer import Population


# -------------------------------------------------------------------
#  SurvivorSelectionDef – direct call
# -------------------------------------------------------------------
def test_survivor_selection_def_calls_wrapped_function():
    def dummy(pop_fit, off_fit, rng):
        return np.array([0, 2])

    def_obj = SurvivorSelectionDef(dummy)
    pop = make_pop([1.0, 2.0], dummy_objfunc)
    off = make_pop([3.0, 4.0], dummy_objfunc)
    result = def_obj(pop, off, random_state=rng)
    assert_array_equal(result, [0, 2])


def test_survivor_selection_def_passes_fitness_and_kwargs():
    captured = {}

    def spy(pop_fit, off_fit, rng, **kw):
        captured["pop_fit"] = pop_fit
        captured["off_fit"] = off_fit
        captured["kw"] = kw
        return np.array([0])

    def_obj = SurvivorSelectionDef(spy, params={"extra": 5})
    pop = make_pop([10.0], dummy_objfunc)
    off = make_pop([20.0], dummy_objfunc)
    def_obj(pop, off, random_state=rng)

    assert_array_equal(captured["pop_fit"], [10.0])
    assert_array_equal(captured["off_fit"], [20.0])
    assert captured["kw"]["extra"] == 5


# -------------------------------------------------------------------
#  create_survivor_selection – type and name
# -------------------------------------------------------------------
@pytest.mark.parametrize("method, expected_type", [
    ("elitism", SurvivorSelectionFromLambda),
    ("generational", SurvivorSelectionFromLambda),
    ("(mu+lambda)", SurvivorSelectionFromLambda),
    ("nothing", NullSurvivorSelection),
])
def test_create_returns_correct_type(method, expected_type, rng):
    sel = create_survivor_selection(method, random_state=rng)
    assert isinstance(sel, expected_type)


def test_create_uses_given_name(rng):
    sel = create_survivor_selection("one_to_one", name="custom_name", random_state=rng)
    assert sel.name == "custom_name"


def test_create_default_name_is_method(rng):
    sel = create_survivor_selection("hillclimb", random_state=rng)
    assert sel.name == "hillclimb"


# -------------------------------------------------------------------
#  Integration: select() returns correct indices (using distinct genotypes)
# -------------------------------------------------------------------
@pytest.mark.parametrize("method, kwargs", [
    ("elitism", {"amount": 1}),
    ("generational", {}),
    ("one_to_one", {}),
    ("(mu+lambda)", {}),
    ("(mu,lambda)", {}),
])
def test_factory_select_returns_valid_survivors(method, kwargs, rng, dummy_objfunc):
    # Build parents and offspring with DIFFERENT genotype matrices
    # so we can identify their origin by genotype content.
    parents = Population(dummy_objfunc, np.array([[1, 2], [3, 4]]))
    parents.fitness = np.array([5.0, 1.0])

    offspring = Population(dummy_objfunc, np.array([[5, 6], [7, 8]]))
    offspring.fitness = np.array([10.0, 2.0])
    offspring.best = offspring.genotype_matrix[0]
    offspring.best_fitness = 10.0

    sel = create_survivor_selection(method, random_state=rng, **kwargs)
    survivors = sel.select(parents, offspring)

    # Must be a Population with same length as parents
    assert len(survivors) == len(parents)

    # All survivors must come from the union of parents and offspring
    full_geno = np.concatenate([parents.genotype_matrix, offspring.genotype_matrix], axis=0)
    for row in survivors.genotype_matrix:
        assert any(np.array_equal(row, full_geno[i]) for i in range(len(full_geno)))

    # Method‑specific checks (now genotype‑based, not index‑based)
    if method == "elitism":
        # Best parent (fitness 5) must survive
        assert np.any(np.all(survivors.genotype_matrix == parents.genotype_matrix[0], axis=1))
    if method == "generational" or method == "(mu,lambda)":
        # No parent genotype should appear in survivors
        for row in survivors.genotype_matrix:
            assert not any(np.array_equal(row, parents.genotype_matrix[i]) for i in range(len(parents)))