"""
Unit tests for parameter schedule implementations.

Contracts verified:
- LinearSchedule interpolates correctly between init and final values.
- LogisticSchedule output is in (0, 1) for progress in (0, 1).
- ExponentialDecaySchedule decreases toward final_value.
- ThresholdSchedule returns first value below threshold, second above.
- RandomSchedule output is within [low, high].
- StepSchedule returns the right value for each progress band.
- All schedules are callable via __call__ and return the same value as evaluate.
"""

import numpy as np
import pytest

from metaheuristic_designer.parameter_schedules import (
    LinearSchedule,
    LogisticSchedule,
    ThresholdSchedule,
    RandomSchedule,
    StepSchedule,
)
from metaheuristic_designer.parameter_schedules.exponential_decay_schedule import (
    ExponentialDecaySchedule,
)


# ---------------------------------------------------------------------------
# LinearSchedule
# ---------------------------------------------------------------------------

def test_linear_schedule_at_progress_0():
    sched = LinearSchedule(init_value=0.0, final_value=10.0)
    assert sched.evaluate(0.0) == pytest.approx(0.0)


def test_linear_schedule_at_progress_1():
    sched = LinearSchedule(init_value=0.0, final_value=10.0)
    assert sched.evaluate(1.0) == pytest.approx(10.0)


def test_linear_schedule_at_midpoint():
    sched = LinearSchedule(init_value=2.0, final_value=8.0)
    assert sched.evaluate(0.5) == pytest.approx(5.0)


def test_linear_schedule_callable():
    sched = LinearSchedule(init_value=0.0, final_value=1.0)
    assert sched(0.25) == sched.evaluate(0.25)


# ---------------------------------------------------------------------------
# LogisticSchedule
# ---------------------------------------------------------------------------

def test_logistic_schedule_output_in_range():
    sched = LogisticSchedule(init_value=0.0, final_value=1.0)
    for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
        val = sched.evaluate(p)
        assert 0.0 <= val <= 1.0


def test_logistic_schedule_monotone():
    sched = LogisticSchedule(init_value=0.0, final_value=1.0)
    values = [sched.evaluate(p) for p in np.linspace(0, 1, 20)]
    assert all(values[i] <= values[i + 1] for i in range(len(values) - 1))


# ---------------------------------------------------------------------------
# ExponentialDecaySchedule
# ---------------------------------------------------------------------------

def test_exponential_decay_iterative_first_step():
    sched = ExponentialDecaySchedule(init_value=10.0, final_value=0.0, alpha=0.5)
    val = sched.evaluate(0.0)
    # 0 + (10 - 0) * 0.5 = 5
    assert val == pytest.approx(5.0)


def test_exponential_decay_iterative_converges():
    sched = ExponentialDecaySchedule(init_value=1.0, final_value=0.0, alpha=0.5)
    val = None
    for _ in range(60):
        val = sched.evaluate(0.0)
    assert val == pytest.approx(0.0, abs=1e-12)


def test_exponential_decay_progress_based_at_0():
    sched = ExponentialDecaySchedule(init_value=10.0, final_value=0.0, alpha=1.0, iterative=False)
    # final + (init - final) * exp(-alpha * 0) = 10
    assert sched.evaluate(0.0) == pytest.approx(10.0)


def test_exponential_decay_progress_based_decreases():
    sched = ExponentialDecaySchedule(init_value=10.0, final_value=0.0, alpha=1.0, iterative=False)
    v0 = sched.evaluate(0.0)
    v1 = sched.evaluate(1.0)
    assert v1 < v0


# ---------------------------------------------------------------------------
# ThresholdSchedule
# API: ThresholdSchedule(init_value, final_value, threshold=0.5)
# Returns init_value when progress < threshold, final_value otherwise.
# ---------------------------------------------------------------------------

def test_threshold_schedule_below_threshold():
    sched = ThresholdSchedule(init_value="low", final_value="high", threshold=0.5)
    assert sched.evaluate(0.3) == "low"


def test_threshold_schedule_above_threshold():
    sched = ThresholdSchedule(init_value="low", final_value="high", threshold=0.5)
    assert sched.evaluate(0.7) == "high"


def test_threshold_schedule_at_threshold():
    # At exactly the threshold: progress < threshold is False → returns final_value
    sched = ThresholdSchedule(init_value="low", final_value="high", threshold=0.5)
    result = sched.evaluate(0.5)
    assert result == "high"


# ---------------------------------------------------------------------------
# RandomSchedule
# API: RandomSchedule(init_value, final_value, random_state=None)
# Returns uniform sample in [init_value, final_value].
# ---------------------------------------------------------------------------

def test_random_schedule_within_bounds():
    sched = RandomSchedule(init_value=2.0, final_value=5.0, random_state=0)
    for p in np.linspace(0, 1, 10):
        val = sched.evaluate(p)
        assert 2.0 <= val <= 5.0


# ---------------------------------------------------------------------------
# StepSchedule
# API: StepSchedule(steps: dict)
# Returns the value corresponding to the last step threshold not exceeded.
# ---------------------------------------------------------------------------

def test_step_schedule_first_band():
    sched = StepSchedule({0.0: "a", 0.5: "b", 0.8: "c"})
    assert sched.evaluate(0.0) == "a"
    assert sched.evaluate(0.4) == "a"


def test_step_schedule_second_band():
    sched = StepSchedule({0.0: "a", 0.5: "b", 0.8: "c"})
    assert sched.evaluate(0.5) == "b"
    assert sched.evaluate(0.79) == "b"


def test_step_schedule_last_band():
    sched = StepSchedule({0.0: "a", 0.5: "b", 0.8: "c"})
    assert sched.evaluate(0.8) == "c"
    assert sched.evaluate(1.0) == "c"


# ---------------------------------------------------------------------------
# LogisticSchedule with exact_bounds=True
# ---------------------------------------------------------------------------

from metaheuristic_designer.parameter_schedules.logistic_schedule import LogisticSchedule


def test_logistic_schedule_exact_bounds_at_0():
    sched = LogisticSchedule(init_value=0.0, final_value=1.0, exact_bounds=True)
    val = sched.evaluate(0.0)
    assert val == pytest.approx(0.0, abs=1e-3)


def test_logistic_schedule_exact_bounds_at_1():
    sched = LogisticSchedule(init_value=0.0, final_value=1.0, exact_bounds=True)
    val = sched.evaluate(1.0)
    assert val == pytest.approx(1.0, abs=1e-3)


def test_logistic_schedule_exact_bounds_midpoint():
    sched = LogisticSchedule(init_value=0.0, final_value=1.0, exact_bounds=True)
    val = sched.evaluate(0.5)
    assert 0.0 < val < 1.0


def test_logistic_schedule_without_exact_bounds():
    sched = LogisticSchedule(init_value=0.0, final_value=1.0, exact_bounds=False)
    val_0 = sched.evaluate(0.0)
    val_1 = sched.evaluate(1.0)
    val_half = sched.evaluate(0.5)
    assert val_0 < val_half < val_1
