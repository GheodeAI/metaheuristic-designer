import numpy as np
from conftest import (
    dummy_objfunc,
    dummy_strategy,
    rng,
)

from metaheuristic_designer.algorithms.standard_algorithm import StandardAlgorithm


def test_standard_algorithm_initialization(dummy_objfunc, dummy_strategy):
    algo = StandardAlgorithm(dummy_objfunc, dummy_strategy, ngen=1, verbose=False)
    assert algo.name == "dummy_strategy"
    assert algo.objfunc is dummy_objfunc
    assert algo.search_strategy is dummy_strategy


def test_standard_algorithm_step_records_history(dummy_objfunc, dummy_strategy):
    algo = StandardAlgorithm(dummy_objfunc, dummy_strategy, ngen=1, neval=1, verbose=False)
    pop = algo.initialize()
    assert len(pop) == dummy_strategy.pop_size

    # Single step
    new_pop = algo.step(population=pop)
    assert len(new_pop) == len(pop)
    # History should contain one fitness record
    assert len(algo.fit_history) == 1
    assert len(algo.best_history) == 1


def test_standard_algorithm_property_delegation(dummy_objfunc, dummy_strategy):
    algo = StandardAlgorithm(dummy_objfunc, dummy_strategy, ngen=1, verbose=False)
    assert algo.iterations == 0
    assert algo.evaluations == 0
    assert algo.patience_left == algo.stopping_condition.patience_left
    assert algo.population is None  # before initialization
    pop = algo.initialize()
    assert algo.population is pop


def test_standard_algorithm_restart(dummy_objfunc, dummy_strategy):
    algo = StandardAlgorithm(dummy_objfunc, dummy_strategy, ngen=1, verbose=False)
    algo.initialize()
    algo.step()
    assert len(algo.fit_history) > 0
    algo.restart()
    assert len(algo.fit_history) == 0