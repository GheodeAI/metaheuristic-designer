import pytest
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from conftest import (
    rng,
    dummy_objfunc,
    dummy_initializer,
    dummy_operator,
    make_pop,
)

from metaheuristic_designer.strategies.classic.hill_climb import HillClimb
from metaheuristic_designer.strategies.classic.local_search import LocalSearch
from metaheuristic_designer.strategies.no_search import NoSearch
from metaheuristic_designer.strategies.static_population_strategy import StaticPopulation
from metaheuristic_designer.strategies.variable_population_strategy import VariablePopulation

from metaheuristic_designer.survivor_selection_base import SurvivorSelection
from metaheuristic_designer.operator import OperatorFromLambda


# ===================================================================
#  HillClimb
# ===================================================================
def test_hill_climb_default_survivor(rng, dummy_initializer):
    algo = HillClimb(initializer=dummy_initializer, random_state=rng)
    assert algo.survivor_sel is not None
    assert isinstance(algo.survivor_sel, SurvivorSelection)
    # default name
    assert algo.name == "HillClimb"


def test_hill_climb_custom_survivor(rng, dummy_initializer):
    from metaheuristic_designer.survivor_selection import create_survivor_selection

    custom_sel = create_survivor_selection("generational", random_state=rng)
    algo = HillClimb(initializer=dummy_initializer, survivor_sel=custom_sel, random_state=rng)
    assert algo.survivor_sel is custom_sel


# ===================================================================
#  LocalSearch
# ===================================================================
def test_local_search_perturb_repeats_parents(rng, dummy_initializer, dummy_objfunc):
    # Use a small initializer for simplicity
    from metaheuristic_designer.initializers import UniformInitializer

    init = UniformInitializer(2, -1, 1, population_size=2, random_state=rng)
    algo = LocalSearch(initializer=init, iterations=3, random_state=rng)
    parents = init.generate_population(dummy_objfunc)
    original_size = len(parents)
    result = algo.perturb(parents)
    # Should repeat parents `iterations` times, then apply operator (which is None/Nothing here)
    # With operator=None, the parent class perturb will just return the repeated population
    # The repetition is done before calling super().perturb, so the new population should be of size pop_size * iterations
    assert len(result) == original_size * 3


# ===================================================================
#  NoSearch
# ===================================================================
def test_no_search_perturb_returns_same_population(rng, dummy_initializer, dummy_objfunc):
    algo = NoSearch(initializer=dummy_initializer, random_state=rng)
    parents = make_pop([1.0, 2.0], dummy_objfunc)
    result = algo.perturb(parents)
    assert result is parents  # same object


# ===================================================================
#  StaticPopulation
# ===================================================================
def test_static_population_requires_operator(rng, dummy_initializer, dummy_operator):
    algo = StaticPopulation(initializer=dummy_initializer, operator=dummy_operator, random_state=rng)
    assert algo.operator is dummy_operator


def test_static_population_accepts_parent_sel(rng, dummy_initializer, dummy_operator):
    from metaheuristic_designer.parent_selection_base import NullParentSelection

    parent_sel = NullParentSelection()
    algo = StaticPopulation(initializer=dummy_initializer, operator=dummy_operator, parent_sel=parent_sel, random_state=rng)
    assert algo.parent_sel is parent_sel


# ===================================================================
#  VariablePopulation
# ===================================================================
def test_variable_population_shuffles_parents(rng, dummy_initializer, dummy_operator, dummy_objfunc):
    algo = VariablePopulation(initializer=dummy_initializer, operator=dummy_operator, random_state=rng)
    parents = dummy_initializer.generate_population(dummy_objfunc)
    assert len(parents) == 10

    shuffled = algo.select_parents(parents)
    assert len(shuffled) == 10

    # Each selected individual must be from the original population (may duplicate)
    for row in shuffled.genotype_matrix:
        assert any(np.array_equal(row, original_row) for original_row in parents.genotype_matrix)


def test_variable_population_custom_offspring_size(rng, dummy_initializer, dummy_operator, dummy_objfunc):
    algo = VariablePopulation(initializer=dummy_initializer, operator=dummy_operator, offspring_size=5, random_state=rng)
    parents = dummy_initializer.generate_population(dummy_objfunc)
    shuffled = algo.select_parents(parents)
    assert len(shuffled) == 5  # custom size


def test_variable_population_initializer_update_changes_offspring_size(rng, dummy_initializer, dummy_operator):
    algo = VariablePopulation(initializer=dummy_initializer, operator=dummy_operator, random_state=rng)
    # Change initializer with a different pop_size
    from metaheuristic_designer.initializers import UniformInitializer

    new_init = UniformInitializer(2, 0, 1, population_size=6, random_state=rng)
    algo.initializer = new_init
    # Since we didn't set custom offspring size originally, it should update offspring_size to 6
    parents = new_init.generate_population(dummy_objfunc)
    shuffled = algo.select_parents(parents)
    assert len(shuffled) == 6
