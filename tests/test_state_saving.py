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

from metaheuristic_designer.algorithms import Algorithm
from metaheuristic_designer.operators import NullOperator
from metaheuristic_designer.parent_selection import NullParentSelection
from metaheuristic_designer.strategies import PopulationBasedStrategy
from metaheuristic_designer.survivor_selection import NullSurvivorSelection


# ===================================================================
#  Population
# ===================================================================
def test_population_get_state():
    pop = make_pop([1.0, 2.0])
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
    assert state["class_name"] == NullOperator.__name__
    assert "name" in state
    assert "encoding" in state


# ===================================================================
#  ParentSelection
# ===================================================================
def test_parent_selection_get_state(dummy_parent_selection):
    state = dummy_parent_selection.get_state()
    assert state["class_name"] == NullParentSelection.__name__
    assert "name" in state
    assert "amount" in state


# ===================================================================
#  SurvivorSelection
# ===================================================================
def test_survivor_selection_get_state(dummy_survivor_selection):
    state = dummy_survivor_selection.get_state()
    assert state["class_name"] == NullSurvivorSelection.__name__
    assert "name" in state


# ===================================================================
#  SearchStrategy
# ===================================================================
def test_search_strategy_get_state(dummy_strategy):
    state = dummy_strategy.get_state()
    assert state["class_name"] == PopulationBasedStrategy.__name__
    assert state["name"] == "dummy_strategy"
    assert "initializer" in state
    assert "operators" in state
    assert len(state["operators"]) >= 1


# ===================================================================
#  Algorithm (Algorithm, initialized)
# ===================================================================
def test_algorithm_get_state(dummy_objfunc, dummy_strategy):
    algo = Algorithm(dummy_objfunc, dummy_strategy, stop_condition_str="max_iterations", max_iterations=1, reporter="silent")
    algo.initialize()  # creates population, evaluates fitness
    state = algo.get_state()
    assert state["name"] == "dummy_strategy"
    assert "objfunc" in state
    assert "search_strategy" in state
    assert "population" in state
    assert "history" in state


# ===================================================================
#  store_state (write to JSON)
# ===================================================================
def test_store_state_to_json(dummy_objfunc, dummy_strategy, tmp_path):
    algo = Algorithm(dummy_objfunc, dummy_strategy, stop_condition_str="max_iterations", max_iterations=1, reporter="silent")
    algo.initialize()
    file_path = tmp_path / "state.json"
    algo.store_state(str(file_path), readable=True)
    # File must be valid JSON
    with open(file_path, "r") as f:
        data = json.load(f)
    assert data["name"] == "dummy_strategy"
    assert "search_strategy" in data
