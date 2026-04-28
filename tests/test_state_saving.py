import json
import numpy as np
from conftest import (
    dummy_objfunc,
    dummy_strategy,
    dummy_initializer,
    dummy_operator,
    dummy_parent_selection,
    dummy_survivor_selection,
    make_pop,
    rng,
)

from metaheuristic_designer.algorithms.standard_algorithm import StandardAlgorithm
from metaheuristic_designer.population import Population
from metaheuristic_designer.search_strategy import SearchStrategy


# ===================================================================
#  Population
# ===================================================================
def test_population_get_state(dummy_objfunc):
    pop = make_pop([1.0, 2.0], dummy_objfunc)
    pop.best = np.array([1.0, 2.0])
    pop.best_fitness = 1.0
    state = pop.get_state()
    assert "genotype_matrix" in state
    assert "fitness" in state
    assert "best" in state
    assert "best_fitness" in state
    assert "encoding" in state


# ===================================================================
#  Operator (via dummy_operator)
# ===================================================================
def test_operator_get_state(dummy_operator):
    state = dummy_operator.get_state()
    assert "name" in state
    assert "encoding" in state
    assert "parameters" in state


# ===================================================================
#  ParentSelection
# ===================================================================
def test_parent_selection_get_state(dummy_parent_selection):
    state = dummy_parent_selection.get_state()
    assert "name" in state
    assert "parameters" in state


# ===================================================================
#  SurvivorSelection
# ===================================================================
def test_survivor_selection_get_state(dummy_survivor_selection):
    state = dummy_survivor_selection.get_state()
    assert "name" in state
    assert "parameters" in state


# ===================================================================
#  SearchStrategy
# ===================================================================
def test_search_strategy_get_state(dummy_strategy):
    state = dummy_strategy.get_state(show_population=False)
    assert state["name"] == "dummy_strategy"
    assert "initializer" in state
    assert "params" in state
    # operator_register should contain at least one operator (the default one)
    assert "operators" in state
    assert len(state["operators"]) >= 1


# ===================================================================
#  Algorithm (StandardAlgorithm, initialized)
# ===================================================================
def test_algorithm_get_state(dummy_objfunc, dummy_strategy):
    algo = StandardAlgorithm(dummy_objfunc, dummy_strategy, ngen=1, verbose=False)
    algo.initialize()  # creates population, evaluates fitness
    state = algo.get_state(show_fit_history=True, show_gen_history=True, show_population=True)
    assert state["name"] == "dummy_strategy"
    assert "objfunc" in state
    assert "generation" in state
    assert "evaluations" in state
    assert "fit_history" in state
    assert "best_history" in state
    assert "search_strategy" in state
    # search_strategy substate should have population data if requested
    assert "population" in state["search_strategy"]


# ===================================================================
#  store_state (write to JSON)
# ===================================================================
def test_store_state_to_json(dummy_objfunc, dummy_strategy, tmp_path):
    algo = StandardAlgorithm(dummy_objfunc, dummy_strategy, ngen=1, verbose=False)
    algo.initialize()
    file_path = tmp_path / "state.json"
    algo.store_state(str(file_path), readable=True, show_fit_history=False, show_gen_history=False, show_population=False)
    # File must be valid JSON
    with open(file_path, "r") as f:
        data = json.load(f)
    assert data["name"] == "dummy_strategy"
    assert "search_strategy" in data