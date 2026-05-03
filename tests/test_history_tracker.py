# tests/test_history_tracker.py
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from conftest import (
    dummy_objfunc,
    dummy_strategy,
    make_pop,
    full_tracker,
    algo_with_full_tracker,
)

from metaheuristic_designer.history_tracker import HistoryTracker


# ----------------------------------------------------------------
# Basic properties and empty state
# ----------------------------------------------------------------
def test_empty_tracker_to_pandas():
    tracker = HistoryTracker(track_best=True)
    df = tracker.to_pandas()
    assert len(df) == 0
    assert list(df.columns) == ["iteration", "best_objective"]


def test_tracker_initial_state(full_tracker):
    assert full_tracker.iterations == 0
    assert full_tracker.best_solutions == []
    assert full_tracker.median_solutions == []
    assert full_tracker.worst_solutions == []
    assert full_tracker.complete_population == []


# ----------------------------------------------------------------
# Recording after manual steps (using algo_with_full_tracker)
# ----------------------------------------------------------------
def test_record_after_one_step(algo_with_full_tracker):
    algo = algo_with_full_tracker
    pop = algo.initialize()

    # Simulate the initial generation record (should be done by optimize)
    algo.history_tracker.step(algo)           # generation 0
    pop = algo.step(population=pop)
    algo.history_tracker.step(algo)           # generation 1

    tracker = algo.history_tracker
    assert tracker.iterations == 2
    assert len(tracker.best_solutions) == 2
    assert len(tracker.best_objective) == 2
    assert len(tracker.complete_population) == 2
    assert len(tracker.median_solutions) == 2
    assert len(tracker.worst_solutions) == 2


def test_best_solution_is_feasible(algo_with_full_tracker):
    algo = algo_with_full_tracker
    algo.initialize()
    algo.history_tracker.step(algo)

    best = algo.history_tracker.best_solutions[-1]
    assert best.shape == (3,)          # vecsize of dummy_objfunc
    assert np.all(best >= 0) and np.all(best <= 1)


def test_to_pandas_output(algo_with_full_tracker):
    algo = algo_with_full_tracker
    algo.initialize()
    algo.history_tracker.step(algo)       # gen 0
    pop = algo.step()                     # gen 1
    algo.history_tracker.step(algo)

    df = algo.history_tracker.to_pandas()
    assert len(df) == 2
    assert "iteration" in df.columns
    assert "best_objective" in df.columns
    assert df["iteration"].tolist() == [0, 1]


def test_get_state(algo_with_full_tracker):
    algo = algo_with_full_tracker
    algo.initialize()
    algo.history_tracker.step(algo)

    state = algo.history_tracker.get_state()
    assert "class_name" in state
    assert "best_solutions" in state
    assert "best_objective" in state
    assert "populations" in state          # from track_complete
    assert "median_solutions" in state
    assert "worst_solutions" in state


def test_restart_clears_all(algo_with_full_tracker):
    algo = algo_with_full_tracker
    algo.initialize()
    algo.history_tracker.step(algo)
    algo.history_tracker.restart()

    tracker = algo.history_tracker
    assert tracker.iterations == 0
    assert tracker.best_solutions == []
    assert tracker.complete_population == []


# ----------------------------------------------------------------
# Correctness: best, median, worst under maximisation
# ----------------------------------------------------------------
def test_correctness_best_median_worst(dummy_objfunc, algo_with_full_tracker):
    """Crafted population with known fitness - must pick correctly."""
    algo = algo_with_full_tracker
    tracker = algo.history_tracker

    # Build a known population
    genotypes = np.array([
        [10, 20],
        [30, 40],
        [50, 60],
        [70, 80],
        [90, 95]
    ], dtype=float)
    pop = make_pop([1.0, 5.0, 2.0, 4.0, 3.0], dummy_objfunc)
    pop.genotype_matrix = genotypes
    pop.fitness = np.array([1.0, 5.0, 2.0, 4.0, 3.0])
    pop.objective = pop.fitness.copy()
    pop.fitness_calculated = np.ones(5, dtype=bool)

    algo.search_strategy.population = pop
    tracker.step(algo)

    # Best: index 1 (fitness 5.0)
    np.testing.assert_array_equal(tracker.best_solutions[0], genotypes[1])
    assert tracker.best_objective[0] == 5.0

    # Median: sorted [1,2,3,4,5] → median 3 → index 4
    np.testing.assert_array_equal(tracker.median_solutions[0], genotypes[4])
    assert tracker.median_objective[0] == 3.0

    # Worst: index 0 (fitness 1.0)
    np.testing.assert_array_equal(tracker.worst_solutions[0], genotypes[0])
    assert tracker.worst_objective[0] == 1.0