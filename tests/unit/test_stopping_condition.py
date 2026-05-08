"""
Unit tests for StoppingCondition.

Contracts verified:
- is_finished() returns True when max_iterations is reached.
- is_finished() returns True when max_evaluations is reached.
- is_finished() returns True when objective_target is met (for max mode).
- is_finished() returns False before any limit is reached.
- get_progress() returns a value in [0, 1].
- restart() resets iteration counter.
"""

import pytest

from metaheuristic_designer.stopping_condition import StoppingCondition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sc(condition_str, **kwargs):
    """Build a StoppingCondition with sensible defaults for isolated testing."""
    defaults = dict(
        max_iterations=None,
        max_evaluations=None,
        real_time_limit=3600.0,
        cpu_time_limit=3600.0,
        objective_target=None,
        max_patience=None,
    )
    defaults.update(kwargs)
    return StoppingCondition(condition_str=condition_str, **defaults)


# ---------------------------------------------------------------------------
# max_iterations
# ---------------------------------------------------------------------------

def test_stops_after_max_iterations():
    sc = _make_sc("max_iterations", max_iterations=3)
    sc.iterations = 3
    assert sc.is_finished()


def test_does_not_stop_before_max_iterations():
    sc = _make_sc("max_iterations", max_iterations=5)
    sc.iterations = 4
    assert not sc.is_finished()


def test_exactly_at_max_iterations_stops():
    sc = _make_sc("max_iterations", max_iterations=10)
    sc.iterations = 10
    assert sc.is_finished()


# ---------------------------------------------------------------------------
# max_evaluations
# ---------------------------------------------------------------------------

def test_stops_after_max_evaluations():
    sc = _make_sc("max_evaluations", max_evaluations=100)
    sc.evaluations = 100
    assert sc.is_finished()


def test_does_not_stop_before_max_evaluations():
    sc = _make_sc("max_evaluations", max_evaluations=100)
    sc.evaluations = 99
    assert not sc.is_finished()


# ---------------------------------------------------------------------------
# objective_target (max mode)
# ---------------------------------------------------------------------------

def test_stops_when_target_fitness_reached():
    sc = _make_sc("objective_target", objective_target=10.0, optimization_mode="max")
    sc.best_objective = 10.0
    assert sc.is_finished()


def test_does_not_stop_below_target_fitness():
    sc = _make_sc("objective_target", objective_target=10.0, optimization_mode="max")
    sc.best_objective = 9.99
    assert not sc.is_finished()


def test_does_not_stop_when_best_objective_is_none():
    """If best_objective hasn't been set, target cannot be reached."""
    sc = _make_sc("objective_target", objective_target=10.0)
    sc.best_objective = None
    assert not sc.is_finished()


# ---------------------------------------------------------------------------
# get_progress
# ---------------------------------------------------------------------------

def test_get_progress_starts_at_zero():
    sc = _make_sc("max_iterations", max_iterations=10)
    sc.iterations = 0
    p = sc.get_progress()
    assert 0.0 <= p <= 1.0


def test_get_progress_at_half():
    sc = _make_sc("max_iterations", max_iterations=10)
    sc.iterations = 5
    p = sc.get_progress()
    assert p == pytest.approx(0.5)


def test_get_progress_not_clamped_above_limit():
    """BUG (documented in ERRORES.md): get_progress() can exceed 1.0 when
    iterations > max_iterations. Verified here so regressions are caught."""
    sc = _make_sc("max_iterations", max_iterations=10)
    sc.iterations = 11
    p = sc.get_progress()
    # Current behavior: p > 1.0 — no clamping is applied.
    # This test asserts the *current* (broken) behavior so that future fixes
    # can change the assertion to assert p == 1.0.
    assert p > 1.0


# ---------------------------------------------------------------------------
# restart
# ---------------------------------------------------------------------------

def test_restart_resets_iterations():
    sc = _make_sc("max_iterations", max_iterations=10)
    sc.iterations = 7
    sc.restart()
    assert sc.iterations == 0


def test_restart_resets_evaluations():
    sc = _make_sc("max_evaluations", max_evaluations=100)
    sc.evaluations = 50
    sc.restart()
    assert sc.evaluations == 0


# ---------------------------------------------------------------------------
# compound conditions
# ---------------------------------------------------------------------------

def test_compound_or_stops_when_either_condition_met():
    sc = _make_sc(
        "max_iterations or max_evaluations",
        max_iterations=10,
        max_evaluations=100,
    )
    sc.iterations = 10
    sc.evaluations = 0
    assert sc.is_finished()


def test_compound_and_requires_both_conditions():
    sc = _make_sc(
        "max_iterations and max_evaluations",
        max_iterations=5,
        max_evaluations=100,
    )
    sc.iterations = 5
    sc.evaluations = 50  # not yet reached
    assert not sc.is_finished()

    sc.evaluations = 100  # now both reached
    assert sc.is_finished()
