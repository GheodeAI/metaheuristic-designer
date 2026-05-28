import numpy as np
from conftest import (
    dummy_objfunc,
    dummy_strategy,
    rng,
)

from metaheuristic_designer.algorithms import Algorithm


def test_standard_algorithm_initialization(dummy_objfunc, dummy_strategy):
    algo = Algorithm(dummy_objfunc, dummy_strategy, stop_condition_str="max_iterations", max_iterations=1, reporter="silent")
    assert algo.name == "dummy_strategy"
    assert algo.objfunc is dummy_objfunc
    assert algo.search_strategy is dummy_strategy


def test_standard_algorithm_step_records_history(dummy_objfunc, dummy_strategy):
    algo = Algorithm(dummy_objfunc, dummy_strategy, stop_condition_str="max_iterations", max_iterations=1, max_evaluations=1, reporter="silent")
    pop = algo.initialize()
    assert len(pop) == dummy_strategy.population_size

    # Single step
    new_pop = algo.step(prev_population=pop)

    assert len(new_pop) == len(pop)
    assert len(algo.history_tracker.best_objective) == 1
    assert len(algo.history_tracker.best_solutions) == 1


def test_standard_algorithm_property_delegation(dummy_objfunc, dummy_strategy):
    algo = Algorithm(dummy_objfunc, dummy_strategy, stop_condition_str="max_iterations", max_iterations=1, reporter="silent")
    assert algo.iterations == 0
    assert algo.evaluations == 0
    assert algo.patience_left == algo.stopping_condition.patience_left
    assert algo.population is None  # before initialization
    pop = algo.initialize()
    assert algo.population is pop


def test_standard_algorithm_restart(dummy_objfunc, dummy_strategy):
    algo = Algorithm(dummy_objfunc, dummy_strategy, stop_condition_str="max_iterations", max_iterations=1, reporter="silent")
    algo.initialize()
    new_pop = algo.update()
    algo.history_tracker.update(algo)
    assert len(algo.history_tracker.best_objective) > 0
    algo.restart()
    assert len(algo.history_tracker.best_objective) == 0
