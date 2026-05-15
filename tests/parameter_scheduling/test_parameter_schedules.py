# tests/test_parameter_schedules.py
import pytest
import numpy as np
from numpy.testing import assert_almost_equal

# All fixtures and mocks come from conftest
from conftest import rng

# Actual classes under test
from metaheuristic_designer.parameter_schedules import (
    LinearSchedule,
    ThresholdSchedule,
    LogisticSchedule,
    RandomSchedule,
    StepSchedule,
)


# ===================================================================
#  LinearSchedule
# ===================================================================
@pytest.mark.parametrize(
    "init, final, progress, expected",
    [
        (0.0, 10.0, 0.0, 0.0),
        (0.0, 10.0, 1.0, 10.0),
        (0.0, 10.0, 0.5, 5.0),
        (5.0, 15.0, 0.2, 7.0),
        (-5.0, 5.0, 0.75, 2.5),
        (2.0, 2.0, 0.3, 2.0),
    ],
)
def test_linear_schedule(init, final, progress, expected):
    sched = LinearSchedule(init_value=init, final_value=final)
    assert sched.evaluate(progress) == pytest.approx(expected)


# ===================================================================
#  ThresholdSchedule
# ===================================================================
@pytest.mark.parametrize(
    "init, final, threshold, progress, expected",
    [
        (0, 100, 0.5, 0.0, 0),
        (0, 100, 0.5, 0.49, 0),
        (0, 100, 0.5, 0.5, 100),
        (0, 100, 0.5, 0.51, 100),
        (0, 100, 0.5, 1.0, 100),
        (10, 20, 0.3, 0.299, 10),
        (10, 20, 0.3, 0.3, 20),
    ],
)
def test_threshold_schedule(init, final, threshold, progress, expected):
    sched = ThresholdSchedule(init_value=init, final_value=final, threshold=threshold)
    assert sched.evaluate(progress) == pytest.approx(expected)


# ===================================================================
#  LogisticSchedule
# ===================================================================
@pytest.mark.parametrize(
    "init, final, k, exact_bounds, progress, expected",
    [
        (0.0, 1.0, 10, False, 0.5, 0.5),
        (0.0, 1.0, 10, True, 0.0, 0.0),
        (0.0, 1.0, 10, True, 1.0, 1.0),
        (10.0, 20.0, 5, False, 0.3, 10 + 10 * (1 / (1 + np.exp(-5 * (0.3 - 0.5))))),
    ],
)
def test_logistic_schedule(init, final, k, exact_bounds, progress, expected):
    sched = LogisticSchedule(init_value=init, final_value=final, k=k, exact_bounds=exact_bounds)
    assert sched.evaluate(progress) == pytest.approx(expected, abs=1e-10)


# ===================================================================
#  RandomSchedule  (uses rng fixture)
# ===================================================================
def test_random_schedule_deterministic_with_seed(rng):
    sched = RandomSchedule(init_value=0.0, final_value=1.0, random_state=rng)
    # First call with seed 42: pre‑computed value
    expected = np.random.default_rng(42).uniform(0.0, 1.0)
    assert sched.evaluate(0.5) == pytest.approx(expected)


def test_random_schedule_values_in_range(rng):
    sched = RandomSchedule(-10, 10, random_state=rng)
    for _ in range(20):
        val = sched.evaluate(0.0)
        assert -10 <= val <= 10


def test_random_schedule_reproducible_with_same_seed(rng):
    # Create two schedules with same seed, verify first call equal
    sched1 = RandomSchedule(5, 15, random_state=rng)
    rng2 = np.random.default_rng(42)  # fresh identical generator
    sched2 = RandomSchedule(5, 15, random_state=rng2)
    assert sched1.evaluate(0.0) == sched2.evaluate(0.0)


# ===================================================================
#  StepSchedule
# ===================================================================
@pytest.fixture
def step_sched():
    # This fixture is defined here, but it's not a class, just a function.
    # Since conftest lacks a StepSchedule fixture and this is a simple helper,
    # we can keep it in the test file – it's not a mock, just a pre‑built schedule.
    return StepSchedule({0.2: "low", 0.5: "mid", 0.8: "high"})


@pytest.mark.parametrize(
    "progress, expected",
    [
        (0.0, "low"),
        (0.19, "low"),
        (0.2, "low"),
        (0.21, "low"),
        (0.499, "low"),
        (0.5, "mid"),
        (0.7, "mid"),
        (0.799, "mid"),
        (0.8, "high"),
        (1.0, "high"),
    ],
)
def test_step_schedule(step_sched, progress, expected):
    assert step_sched.evaluate(progress) == expected


def test_step_schedule_single_key():
    sched = StepSchedule({0.0: "start"})
    assert sched.evaluate(0.5) == "start"


def test_step_schedule_empty_steps():
    with pytest.raises(IndexError):
        sched = StepSchedule({})
        sched.evaluate(0.5)
