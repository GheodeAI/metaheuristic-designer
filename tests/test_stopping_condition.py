import pytest
from metaheuristic_designer.stopping_condition import (
    StoppingCondition,
    parse_stopping_cond,
    process_condition,
    process_progress,
)


# -------------------------------------------------------------------
#  parse_stopping_cond
# -------------------------------------------------------------------
def test_parse_simple():
    assert parse_stopping_cond("max_iterations") == ["max_iterations"]


def test_parse_or():
    assert parse_stopping_cond("max_evaluations or real_time_limit") == [["max_evaluations", "or", "real_time_limit"]]


def test_parse_and():
    assert parse_stopping_cond("max_iterations and objective_target") == [["max_iterations", "and", "objective_target"]]


def test_parse_complex():
    result = parse_stopping_cond("max_evaluations or max_iterations and real_time_limit")
    expected = [[["max_evaluations", "or", "max_iterations"], "and", "real_time_limit"]]
    assert result == expected


# -------------------------------------------------------------------
#  process_condition
# -------------------------------------------------------------------
def test_process_condition_max_evaluations_true():
    assert process_condition(["max_evaluations"], True, False, False, False, False, False) is True


def test_process_condition_or():
    cond = [["max_evaluations", "or", "real_time_limit"]]
    assert process_condition(cond, True, False, False, False, False, False) is True


def test_process_condition_and():
    cond = [["max_iterations", "and", "objective_target"]]
    assert process_condition(cond, False, True, False, False, True, False) is True
    assert process_condition(cond, False, True, False, False, False, False) is False


# -------------------------------------------------------------------
#  process_progress
# -------------------------------------------------------------------
def test_process_progress_single():
    assert process_progress(["max_iterations"], 0.5, 0.0, 0.0, 0.0, 0.0, 0.0) == 0.0


def test_process_progress_or():
    cond = [["max_evaluations", "or", "real_time_limit"]]
    assert process_progress(cond, 0.3, 0.0, 0.8, 0.0, 0.0, 0.0) == 0.8


def test_process_progress_and():
    cond = [["max_iterations", "and", "objective_target"]]
    assert process_progress(cond, 0.0, 0.7, 0.0, 0.0, 0.9, 0.0) == 0.7


# -------------------------------------------------------------------
#  StoppingCondition high‑level
# -------------------------------------------------------------------
def test_stopping_condition_restart_resets_counters():
    sc = StoppingCondition(condition_str="max_evaluations", max_evaluations=10)
    sc.iterations = 42
    sc.evaluations = 100
    sc.restart()
    assert sc.iterations == 0
    assert sc.evaluations == 0


def test_stopping_condition_is_finished_max_evaluations():
    sc = StoppingCondition(condition_str="max_evaluations", max_evaluations=10)
    sc.evaluations = 10
    sc.best_objective = 0.0
    sc.real_time_spent = 0.0
    sc.cpu_time_spent = 0.0
    assert sc.is_finished() is True


def test_stopping_condition_is_finished_false():
    sc = StoppingCondition(condition_str="max_iterations", max_iterations=100)
    sc.iterations = 50
    sc.best_objective = 0.0
    sc.real_time_spent = 0.0
    sc.cpu_time_spent = 0.0
    assert sc.is_finished() is False
