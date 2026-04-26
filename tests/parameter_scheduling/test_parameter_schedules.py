import pytest
import numpy as np
from metaheuristic_designer.parameter_schedules.linear_schedule import LinearSchedule
from metaheuristic_designer.parameter_schedules.random_schedule import RandomSchedule
from metaheuristic_designer.parameter_schedules.threshold_schedule import ThresholdSchedule
from metaheuristic_designer.parameter_schedules.step_schedule import StepSchedule
from metaheuristic_designer.parameter_schedules.logistic_schedule import LogisticSchedule


# ============================= LinearSchedule =============================
def test_linear_schedule_start_and_end():
    sched = LinearSchedule(init_value=0.0, final_value=10.0)
    assert sched.evaluate(0.0) == 0.0
    assert sched.evaluate(1.0) == 10.0

def test_linear_schedule_midpoint():
    sched = LinearSchedule(init_value=2.0, final_value=8.0)
    assert sched.evaluate(0.5) == 5.0

def test_linear_schedule_decreasing():
    sched = LinearSchedule(init_value=10.0, final_value=0.0)
    assert sched.evaluate(0.0) == 10.0
    assert sched.evaluate(1.0) == 0.0
    assert sched.evaluate(0.2) == 8.0


# ============================= RandomSchedule =============================
def test_random_schedule_uses_bounds():
    sched = RandomSchedule(init_value=1.0, final_value=2.0, random_state=42)
    for _ in range(20):
        val = sched.evaluate(0.3)   # progress ignored by RandomSchedule
        assert 1.0 <= val <= 2.0

def test_random_schedule_reproducible():
    s1 = RandomSchedule(0, 1, random_state=42)
    s2 = RandomSchedule(0, 1, random_state=42)
    assert s1.evaluate(0.3) == s2.evaluate(0.3)


# ============================= ThresholdSchedule =============================
def test_threshold_basic():
    sched = ThresholdSchedule(threshold=0.3, init_value=10.0, final_value=20.0)
    assert sched.evaluate(0.0) == 10.0
    assert sched.evaluate(0.2) == 10.0
    assert sched.evaluate(0.3) == 20.0 
    assert sched.evaluate(0.5) == 20.0
    assert sched.evaluate(1.0) == 20.0


def test_threshold_exact_boundary():
    sched = ThresholdSchedule(threshold=0.5, init_value=0, final_value=1)
    assert sched.evaluate(0.5) == 1


def test_threshold_negative():
    sched = ThresholdSchedule(threshold=0.0, init_value=-5, final_value=10)
    assert sched.evaluate(-0.1) == -5
    assert sched.evaluate(0.0) == 10
    assert sched.evaluate(1.0) == 10


def test_threshold_callable():
    sched = ThresholdSchedule(threshold=0.7, init_value=100, final_value=200)
    assert sched(0.6) == 100
    assert sched(0.8) == 200


# ============================= StepSchedule =============================
def test_step_schedule_single_step():
    sched = StepSchedule(steps={0.0: 1.0, 0.5: 2.0, 1.0: 5.0})
    assert sched.evaluate(0.0) == 1.0
    assert sched.evaluate(0.2) == 1.0
    assert sched.evaluate(0.5) == 2.0
    assert sched.evaluate(0.7) == 2.0
    assert sched.evaluate(1.0) == 5.0

def test_step_schedule_out_of_order_keys():
    sched = StepSchedule(steps={0.5: 2.0, 0.0: 1.0, 0.8: 3.0})
    assert sched.evaluate(0.0) == 1.0
    assert sched.evaluate(0.6) == 2.0
    assert sched.evaluate(0.9) == 3.0


# ============================= LogisticSchedule ===========================
def test_logistic_schedule_exact_bounds():
    """When exact_bounds=True, endpoints are exactly init and final."""
    sched = LogisticSchedule(init_value=0.0, final_value=1.0, k=10.0, exact_bounds=True)
    assert sched.evaluate(0.0) == pytest.approx(0.0, abs=1e-6)
    assert sched.evaluate(1.0) == pytest.approx(1.0, abs=1e-6)
    # Midpoint should still be near the middle
    mid = sched.evaluate(0.5)
    assert mid == pytest.approx(0.5, abs=0.05)

def test_logistic_schedule_no_exact_bounds():
    """Without exact bounds, endpoints are only asymptotically approached."""
    sched = LogisticSchedule(init_value=0.0, final_value=1.0, k=10.0, exact_bounds=False)
    # With k=10, f(0) ≈ 0.0067
    assert sched.evaluate(0.0) == pytest.approx(0.0, abs=0.01)
    assert sched.evaluate(1.0) == pytest.approx(1.0, abs=0.01)