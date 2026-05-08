"""
Unit tests for HistoryTracker.

Contract:
- step() records iteration data for all enabled tracking modes.
- to_pandas() returns a DataFrame with the tracked columns.
- restart() clears all history.
- Diversity, worst, median, full_objective, full_population tracking works.
"""

import numpy as np
import pytest

from metaheuristic_designer.history_tracker import HistoryTracker
from metaheuristic_designer.benchmarks import Sphere
from metaheuristic_designer import simple


COMMON_BASE = {
    "stop_cond": "max_iterations",
    "max_iterations": 5,
    "reporter": "silent",
}


def _run_with_tracker(tracker, extra_kwargs=None):
    """Run a simple algorithm with a given history tracker."""
    objfunc = Sphere(dimension=4, mode="min")
    kwargs = dict(**COMMON_BASE)
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    algo = simple.hill_climb_real(objfunc, random_state=0,
                                   history_tracker=tracker, **kwargs)
    algo.optimize()
    return algo


# ---------------------------------------------------------------------------
# Basic tracking
# ---------------------------------------------------------------------------

def test_tracker_records_best_by_default():
    tracker = HistoryTracker(track_best=True)
    _run_with_tracker(tracker)
    assert len(tracker.best_objective) > 0
    assert len(tracker.best_solutions) > 0


def test_tracker_best_objective_all_finite():
    tracker = HistoryTracker(track_best=True)
    _run_with_tracker(tracker)
    assert all(np.isfinite(v) for v in tracker.best_objective)


def test_tracker_to_pandas_has_iteration_column():
    tracker = HistoryTracker(track_best=True)
    _run_with_tracker(tracker)
    df = tracker.to_pandas()
    assert "iteration" in df.columns


def test_tracker_to_pandas_has_best_objective_column():
    tracker = HistoryTracker(track_best=True)
    _run_with_tracker(tracker)
    df = tracker.to_pandas()
    assert "best_objective" in df.columns


# ---------------------------------------------------------------------------
# Median tracking
# ---------------------------------------------------------------------------

def test_tracker_median_recording():
    tracker = HistoryTracker(track_best=True, track_median=True)
    _run_with_tracker(tracker)
    assert len(tracker.median_objective) > 0


def test_tracker_to_pandas_median_column():
    tracker = HistoryTracker(track_best=True, track_median=True)
    _run_with_tracker(tracker)
    df = tracker.to_pandas()
    assert "median_objective" in df.columns


# ---------------------------------------------------------------------------
# Worst tracking
# ---------------------------------------------------------------------------

def test_tracker_worst_recording():
    tracker = HistoryTracker(track_best=True, track_worst=True)
    _run_with_tracker(tracker)
    assert len(tracker.worst_objective) > 0


def test_tracker_to_pandas_worst_column():
    tracker = HistoryTracker(track_best=True, track_worst=True)
    _run_with_tracker(tracker)
    df = tracker.to_pandas()
    assert "worst_objective" in df.columns


# ---------------------------------------------------------------------------
# Diversity tracking
# ---------------------------------------------------------------------------

def test_tracker_diversity_recording():
    tracker = HistoryTracker(track_best=True, track_diversity=True)
    _run_with_tracker(tracker)
    assert len(tracker.diversity) > 0


def test_tracker_to_pandas_diversity_column():
    tracker = HistoryTracker(track_best=True, track_diversity=True)
    _run_with_tracker(tracker)
    df = tracker.to_pandas()
    assert "diversity" in df.columns


# ---------------------------------------------------------------------------
# Full objective and population tracking
# ---------------------------------------------------------------------------

def test_tracker_full_objective_recording():
    tracker = HistoryTracker(track_best=True, track_full_objective=True)
    _run_with_tracker(tracker)
    assert len(tracker.complete_objective) > 0


def test_tracker_full_population_recording():
    tracker = HistoryTracker(track_best=True, track_full_population=True)
    _run_with_tracker(tracker)
    assert len(tracker.complete_population) > 0


# ---------------------------------------------------------------------------
# restart()
# ---------------------------------------------------------------------------

def test_tracker_restart_clears_history():
    tracker = HistoryTracker(track_best=True, track_median=True, track_worst=True,
                              track_diversity=True)
    _run_with_tracker(tracker)
    assert len(tracker.best_objective) > 0
    tracker.restart()
    assert len(tracker.best_objective) == 0
    assert len(tracker.median_objective) == 0
    assert len(tracker.worst_objective) == 0
    assert len(tracker.diversity) == 0
    assert len(tracker.recorded_iterations) == 0


# ---------------------------------------------------------------------------
# to_pandas_full_objective
# ---------------------------------------------------------------------------

def test_tracker_to_pandas_full_objective_when_tracked():
    tracker = HistoryTracker(track_full_objective=True)
    _run_with_tracker(tracker)
    df = tracker.to_pandas_full_objective()
    assert df is not None


def test_tracker_to_pandas_full_objective_when_not_tracked_returns_empty():
    tracker = HistoryTracker(track_full_objective=False)
    _run_with_tracker(tracker)
    df = tracker.to_pandas_full_objective()
    assert df.empty


# ---------------------------------------------------------------------------
# All tracking flags enabled together
# ---------------------------------------------------------------------------

def test_tracker_all_flags_enabled():
    tracker = HistoryTracker(
        track_best=True,
        track_median=True,
        track_worst=True,
        track_full_objective=True,
        track_full_population=True,
        track_diversity=True,
    )
    _run_with_tracker(tracker)
    assert len(tracker.best_objective) > 0
    assert len(tracker.median_objective) > 0
    assert len(tracker.worst_objective) > 0
    assert len(tracker.complete_objective) > 0
    assert len(tracker.complete_population) > 0
    assert len(tracker.diversity) > 0

    df = tracker.to_pandas()
    assert "best_objective" in df.columns
    assert "median_objective" in df.columns
    assert "worst_objective" in df.columns
    assert "diversity" in df.columns
    assert len(df) > 0


# ---------------------------------------------------------------------------
# to_pandas when no best tracking
# ---------------------------------------------------------------------------

def test_tracker_to_pandas_without_best_only_has_iteration():
    tracker = HistoryTracker(track_best=False)
    _run_with_tracker(tracker)
    df = tracker.to_pandas()
    assert "iteration" in df.columns
    assert "best_objective" not in df.columns
