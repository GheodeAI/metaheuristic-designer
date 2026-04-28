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
    assert parse_stopping_cond("ngen") == ["ngen"]

def test_parse_or():
    assert parse_stopping_cond("neval or time_limit") == [["neval", "or", "time_limit"]]

def test_parse_and():
    assert parse_stopping_cond("ngen and fit_target") == [["ngen", "and", "fit_target"]]

def test_parse_complex():
    result = parse_stopping_cond("neval or ngen and time_limit")
    # The parser returns [[['neval','or','ngen'], 'and', 'time_limit']]
    expected = [[["neval", "or", "ngen"], "and", "time_limit"]]
    assert result == expected


# -------------------------------------------------------------------
#  process_condition
# -------------------------------------------------------------------
def test_process_condition_neval_true():
    assert process_condition(["neval"], True, False, False, False, False, False) is True

def test_process_condition_or():
    cond = [["neval", "or", "time_limit"]]
    assert process_condition(cond, True, False, False, False, False, False) is True

def test_process_condition_and():
    cond = [["ngen", "and", "fit_target"]]
    assert process_condition(cond, False, True, False, False, True, False) is True
    assert process_condition(cond, False, True, False, False, False, False) is False


# -------------------------------------------------------------------
#  process_progress
# -------------------------------------------------------------------
def test_process_progress_single():
    assert process_progress(["ngen"], 0.5, 0.0, 0.0, 0.0, 0.0, 0.0) == 0.0

def test_process_progress_or():
    cond = [["neval", "or", "time_limit"]]
    assert process_progress(cond, 0.3, 0.0, 0.8, 0.0, 0.0, 0.0) == 0.8

def test_process_progress_and():
    cond = [["ngen", "and", "fit_target"]]
    assert process_progress(cond, 0.0, 0.7, 0.0, 0.0, 0.9, 0.0) == 0.7


# -------------------------------------------------------------------
#  StoppingCondition high‑level
# -------------------------------------------------------------------
def test_stopping_condition_defaults():
    sc = StoppingCondition()
    assert sc.max_iterations == 1000
    assert sc.time_limit == 60.0

def test_stopping_condition_restart_resets_counters():
    sc = StoppingCondition()
    sc.iterations = 42
    sc.evaluations = 100
    sc.restart()
    assert sc.iterations == 0
    assert sc.evaluations == 0

def test_stopping_condition_is_finished_neval():
    sc = StoppingCondition(condition_str="neval", max_evaluations=10)
    sc.evaluations = 10
    sc.best_fitness = 0.0
    sc.real_time_spent = 0.0
    sc.cpu_time_spent = 0.0
    assert sc.is_finished() is True

def test_stopping_condition_is_finished_false():
    sc = StoppingCondition(condition_str="ngen", max_iterations=100)
    sc.iterations = 50
    sc.best_fitness = 0.0
    sc.real_time_spent = 0.0
    sc.cpu_time_spent = 0.0
    assert sc.is_finished() is False