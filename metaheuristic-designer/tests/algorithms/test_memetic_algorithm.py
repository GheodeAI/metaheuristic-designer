# tests/algorithms/test_memetic_algorithm.py
import logging
import numpy as np
from numpy.testing import assert_array_equal

from conftest import (
    rng,
    dummy_objfunc,
    dummy_strategy,
    dummy_initializer,
    dummy_parent_selection,
)

from metaheuristic_designer.algorithms.memetic_algorithm import MemeticAlgorithm
from metaheuristic_designer.survivor_selection import create_survivor_selection
from metaheuristic_designer.operators import create_operator
from metaheuristic_designer.search_strategy import SearchStrategy
from metaheuristic_designer.parent_selection_base import NullParentSelection
from metaheuristic_designer.operator import OperatorFromLambda


# ===================================================================
#  Construction & name
# ===================================================================
def test_memetic_algorithm_creation(dummy_objfunc, dummy_strategy, dummy_parent_selection):
    algo = MemeticAlgorithm(
        dummy_objfunc,
        search_strategy=dummy_strategy,
        local_search=dummy_strategy,
        improvement_selection=dummy_parent_selection,
        max_iterations=1,
        max_evaluations=50,
        reporter="silent",
        stop_cond="max_iterations",
    )
    assert algo.name == "Memetic dummy_strategy"
    assert algo.local_search is dummy_strategy
    assert algo.improvement_selection is dummy_parent_selection
    assert algo.keep_improved_solutions is True
    assert algo.local_search_frequency == 1
    assert algo.local_search_depth == 1


def test_memetic_algorithm_custom_name(dummy_objfunc, dummy_strategy, dummy_parent_selection):
    algo = MemeticAlgorithm(
        dummy_objfunc,
        dummy_strategy,
        local_search=dummy_strategy,
        improvement_selection=dummy_parent_selection,
        name="CustomMemetic",
        max_iterations=1,
        max_evaluations=50,
        reporter="silent",
        stop_cond="max_iterations",
    )
    assert algo.name == "CustomMemetic"


# ===================================================================
#  initialize
# ===================================================================
def test_initialize_calls_local_search_initialize(dummy_objfunc, dummy_strategy, dummy_initializer, dummy_parent_selection):
    local_search = SearchStrategy(initializer=dummy_initializer, name="local_searcher")
    algo = MemeticAlgorithm(
        dummy_objfunc,
        dummy_strategy,
        local_search=local_search,
        improvement_selection=dummy_parent_selection,
        max_iterations=1,
        max_evaluations=50,
        reporter="silent",
        stop_cond="max_iterations",
    )
    pop = algo.initialize()
    assert local_search.population is not None
    assert len(local_search.population) == dummy_initializer.population_size


# ===================================================================
#  step (Lamarckian, default)
# ===================================================================
def test_step_lamarckian_records_history(dummy_objfunc, dummy_strategy, dummy_initializer, rng):
    operator = create_operator("nothing", random_state=rng)
    survivor_sel = create_survivor_selection("one_to_one", random_state=rng)
    local_search = SearchStrategy(
        initializer=dummy_initializer,
        operator=operator,
        survivor_sel=survivor_sel,
        name="local_searcher",
    )
    improvement_sel = NullParentSelection()

    algo = MemeticAlgorithm(
        dummy_objfunc,
        dummy_strategy,
        local_search=local_search,
        improvement_selection=improvement_sel,
        max_iterations=1,
        max_evaluations=50,
        reporter="silent",
        stop_cond="max_iterations",
        keep_improved_solutions=True,
    )
    algo.initialize()
    _ = algo.step()
    algo.history_tracker.step(algo)
    assert len(algo.history_tracker.best_objective) == 1
    assert len(algo.history_tracker.best_solutions) == 1


# ===================================================================
#  Baldwinian mode
# ===================================================================
def test_baldwinian_does_not_change_genotype(dummy_objfunc, dummy_strategy, dummy_initializer, rng):
    operator = create_operator("nothing", random_state=rng)
    survivor_sel = create_survivor_selection("one_to_one", random_state=rng)
    local_search = SearchStrategy(
        initializer=dummy_initializer,
        operator=operator,
        survivor_sel=survivor_sel,
        name="local_searcher",
    )
    improvement_sel = NullParentSelection()

    algo = MemeticAlgorithm(
        dummy_objfunc,
        dummy_strategy,
        local_search=local_search,
        improvement_selection=improvement_sel,
        max_iterations=1,
        max_evaluations=50,
        reporter="silent",
        stop_cond="max_iterations",
        keep_improved_solutions=False,
    )
    algo.initialize()
    original_geno = algo.search_strategy.population.genotype_matrix.copy()
    algo.step()
    assert_array_equal(algo.search_strategy.population.genotype_matrix, original_geno)


# ===================================================================
#  local_search_frequency
# ===================================================================
def test_local_search_frequency_skips(dummy_objfunc, dummy_strategy, dummy_initializer, rng):
    operator = create_operator("nothing", random_state=rng)
    survivor_sel = create_survivor_selection("one_to_one", random_state=rng)
    local_search = SearchStrategy(
        initializer=dummy_initializer,
        operator=operator,
        survivor_sel=survivor_sel,
        name="local_searcher",
    )
    algo = MemeticAlgorithm(
        dummy_objfunc,
        dummy_strategy,
        local_search=local_search,
        improvement_selection=NullParentSelection(),
        max_iterations=1,
        max_evaluations=50,
        reporter="silent",
        stop_cond="max_iterations",
        local_search_frequency=2,
    )
    algo.initialize()
    algo.step()


# ===================================================================
#  local_search_depth
# ===================================================================
def test_local_search_depth_multiple(dummy_objfunc, dummy_strategy, dummy_initializer, rng):
    def add_one(pop, init, rng, **kw):
        return pop.update_genotype(pop.genotype_matrix + 1)

    operator = OperatorFromLambda(add_one, preserves_order=True, random_state=rng)
    survivor_sel = create_survivor_selection("one_to_one", random_state=rng)
    local_search = SearchStrategy(
        initializer=dummy_initializer,
        operator=operator,
        survivor_sel=survivor_sel,
        name="local_searcher",
    )
    algo = MemeticAlgorithm(
        dummy_objfunc,
        dummy_strategy,
        local_search=local_search,
        improvement_selection=NullParentSelection(),
        max_iterations=1,
        max_evaluations=50,
        reporter="silent",
        stop_cond="max_iterations",
        local_search_depth=2,
    )
    algo.initialize()
    orig = algo.search_strategy.population.genotype_matrix.copy()
    algo.step()
    expected = orig + 2
    assert_array_equal(algo.search_strategy.population.genotype_matrix, expected)


# ===================================================================
#  Order‑preservation warning (via logging)
# ===================================================================
def test_order_preservation_warning(dummy_objfunc, dummy_strategy, dummy_initializer, rng, caplog):
    operator = create_operator("nothing", random_state=rng)
    survivor_sel = create_survivor_selection("elitism", random_state=rng)  # NOT order‑preserving
    local_search = SearchStrategy(
        initializer=dummy_initializer,
        operator=operator,
        survivor_sel=survivor_sel,
        name="local_searcher",
    )

    with caplog.at_level(logging.WARNING, logger="metaheuristic_designer.algorithms.memetic_algorithm"):
        algo = MemeticAlgorithm(
            dummy_objfunc,
            dummy_strategy,
            local_search=local_search,
            improvement_selection=NullParentSelection(),
            max_iterations=1,
            max_evaluations=50,
            reporter="silent",
            stop_cond="max_iterations",
        )

    # Check that at least one warning was logged (we don't require exact text)
    assert len(caplog.records) >= 1


def test_no_warning_when_order_preserved(dummy_objfunc, dummy_strategy, dummy_initializer, rng, caplog):
    operator = create_operator("nothing", random_state=rng)
    survivor_sel = create_survivor_selection("one_to_one", random_state=rng)
    local_search = SearchStrategy(
        initializer=dummy_initializer,
        operator=operator,
        survivor_sel=survivor_sel,
        name="local_searcher",
    )

    with caplog.at_level(logging.WARNING, logger="metaheuristic_designer.algorithms.memetic_algorithm"):
        MemeticAlgorithm(
            dummy_objfunc,
            dummy_strategy,
            local_search=local_search,
            improvement_selection=NullParentSelection(),
            max_iterations=1,
            max_evaluations=50,
            reporter="silent",
            stop_cond="max_iterations",
        )

    # No warnings should be logged
    assert len(caplog.records) == 0
