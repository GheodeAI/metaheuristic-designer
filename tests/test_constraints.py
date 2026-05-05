import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal

# conftest helper – we need the dummy extended encoding
from conftest import DummyParameterExtendingEncoding

# concrete handlers
from metaheuristic_designer.constraint_handlers import (
    ClipBoundConstraint,
    BounceBoundConstraint,
    CycleBoundConstraint,
    LinearBoundPenaltyConstraint,
    CompositeConstraint,
    ExtendedConstraintHandler,
)
from metaheuristic_designer.constraint_handler import (
    ConstraintHandlerFromLambda,
    NullConstraint,
)


# ===================================================================
#  NullConstraint
# ===================================================================
def test_null_repair_does_nothing():
    handler = NullConstraint()
    orig = np.array([1.0, -5.0])
    repaired = handler.repair_solution(orig)
    assert_array_equal(repaired, orig)
    assert repaired is not orig  # copy


def test_null_penalty_zero():
    handler = NullConstraint()
    np.testing.assert_almost_equal(handler.penalty(np.array([100.0, -200.0])), 0.0)


# ===================================================================
#  ConstraintHandlerFromLambda
# ===================================================================
def test_lambda_repair_calls_function():
    handler = ConstraintHandlerFromLambda(repair_solution_fn=lambda x: x * 2)
    assert_array_equal(handler.repair_solution(np.array([1, 2])), [2, 4])


def test_lambda_penalty_calls_function():
    handler = ConstraintHandlerFromLambda(penalty_fn=lambda x: 10.0)
    assert handler.penalty(np.array([0])) == 10.0


def test_lambda_missing_both_raises():
    with pytest.raises(ValueError):
        ConstraintHandlerFromLambda()


# ===================================================================
#  ClipBoundConstraint
# ===================================================================
@pytest.mark.parametrize(
    "low, high, solution, expected",
    [
        (-1.0, 1.0, np.array([0.0, -0.5, 0.5, 2.0, -3.0]), np.array([0.0, -0.5, 0.5, 1.0, -1.0])),
        (0.0, 5.0, np.array([-2, 0, 3, 6]), np.array([0, 0, 3, 5])),
        (-5.0, -1.0, np.array([-10, -3, 0]), np.array([-5, -3, -1])),
        (0.0, 1.0, np.array([0.5]), np.array([0.5])),
    ],
)
def test_clip_bound_repair(low, high, solution, expected):
    handler = ClipBoundConstraint(vecsize=len(solution), lower_bound=low, upper_bound=high)
    repaired = handler.repair_solution(solution)
    assert_array_equal(repaired, expected)


def test_clip_bound_penalty_zero():
    handler = ClipBoundConstraint(vecsize=2, lower_bound=0, upper_bound=1)
    assert handler.penalty(np.array([0.5, 0.5])) == 0.0


# ===================================================================
#  BounceBoundConstraint
# ===================================================================
@pytest.mark.parametrize(
    "low, high, solution, expected",
    [
        (0.0, 4.0, np.array([1.0, -2.0, 6.0]), np.array([1.0, 2.0, 2.0])),  # -2 bounces to 2, 6 bounces to 2
        (0.0, 10.0, np.array([-5, 15]), np.array([5, 5])),  # -5 -> 5, 15 -> 5
        (0.0, 1.0, np.array([-0.2, 1.3]), np.array([0.2, 0.7])),  # bounce inside unit
    ],
)
def test_bounce_bound_repair(low, high, solution, expected):
    handler = BounceBoundConstraint(vecsize=len(solution), lower_bound=low, upper_bound=high)
    repaired = handler.repair_solution(solution)
    assert_array_equal(repaired, expected)


def test_bounce_bound_penalty_zero():
    handler = BounceBoundConstraint(vecsize=3)
    assert handler.penalty(np.array([0.0, 50.0, 200.0])) == 0.0


# This test will fail until the code returns an array when bounds are equal
def test_bounce_bound_equal_lims_returns_array():
    handler = BounceBoundConstraint(vecsize=3, lower_bound=5.0, upper_bound=5.0)
    solution = np.array([1.0, 2.0, 3.0])
    result = handler.repair_solution(solution)
    # Should return an array filled with 5.0, not a scalar
    assert isinstance(result, np.ndarray)
    assert result.shape == solution.shape
    assert_array_equal(result, np.full(solution.shape, 5.0))


# ===================================================================
#  CycleBoundConstraint
# ===================================================================
@pytest.mark.parametrize(
    "low, high, solution, expected",
    [
        (0.0, 4.0, np.array([1.0, -2.0, 6.0]), np.array([1.0, 2.0, 2.0])),  # -2 wraps to 2, 6 wraps to 2
        (0.0, 10.0, np.array([-5, 15]), np.array([5, 5])),
        (0.0, 1.0, np.array([-0.2, 1.3]), np.array([0.8, 0.3])),  # purely periodic
    ],
)
def test_cycle_bound_repair(low, high, solution, expected):
    handler = CycleBoundConstraint(vecsize=len(solution), lower_bound=low, upper_bound=high)
    repaired = handler.repair_solution(solution)
    np.testing.assert_allclose(repaired, expected)


def test_cycle_bound_penalty_zero():
    handler = CycleBoundConstraint(vecsize=4)
    assert handler.penalty(np.array([10.0, -10.0, 0.0, 100.0])) == 0.0


# Same potential bug as bounce: equal limits should return an array
def test_cycle_bound_equal_lims_returns_array():
    handler = CycleBoundConstraint(vecsize=2, lower_bound=7.0, upper_bound=7.0)
    solution = np.array([99.0, -99.0])
    result = handler.repair_solution(solution)
    assert isinstance(result, np.ndarray)
    assert result.shape == solution.shape
    assert_array_equal(result, np.full(solution.shape, 7.0))


# ===================================================================
#  LinearBoundPenaltyConstraint
# ===================================================================
@pytest.mark.parametrize(
    "low, high, alpha, solution, expected_penalty",
    [
        (0.0, 1.0, 1.0, np.array([[0.2, -0.3, 1.5]]), 0.8),
        (-2.0, 2.0, 2.0, np.array([[-3.0, 0.0, 3.0]]), 4.0),
        (0.0, 10.0, 1.0, np.array([[5.0, 0.0, 10.0]]), 0.0),
    ],
)
def test_linear_bound_penalty(low, high, alpha, solution, expected_penalty):
    handler = LinearBoundPenaltyConstraint(vecsize=solution.shape[1], lower_bound=low, upper_bound=high, alpha=alpha)
    penalty = handler.penalty(solution)  # shape (1,)
    assert penalty.shape == (1,)
    np.testing.assert_allclose(penalty[0], expected_penalty)


def test_linear_bound_repair_does_nothing():
    handler = LinearBoundPenaltyConstraint(vecsize=3, lower_bound=-1, upper_bound=1)
    orig = np.array([0.5, -0.2, 1.2])
    repaired = handler.repair_solution(orig)
    assert_array_equal(repaired, orig)  # it's a PenalizeConstraint, repair just copies


# ===================================================================
#  CompositeConstraint
# ===================================================================
def test_composite_repair_applies_in_order():
    # First handler clips to [-2,2], second clips to [-1,3]
    h1 = ClipBoundConstraint(vecsize=3, lower_bound=-2, upper_bound=2)
    h2 = ClipBoundConstraint(vecsize=3, lower_bound=-1, upper_bound=3)
    comp = CompositeConstraint([h1, h2])
    solution = np.array([-5.0, 0.0, 4.0])
    repaired = comp.repair_solution(solution)
    # After h1: [-2, 0, 2]; after h2: clip to [-1,3] -> [-1, 0, 2]
    expected = np.array([-1.0, 0.0, 2.0])
    assert_array_equal(repaired, expected)


def test_composite_penalty_sums():
    # Two penalty handlers: one returns 2, another returns 3
    p1 = ConstraintHandlerFromLambda(penalty_fn=lambda x: 2.0)
    p2 = ConstraintHandlerFromLambda(penalty_fn=lambda x: 3.0)
    comp = CompositeConstraint([p1, p2])
    assert comp.penalty(np.array([0])) == 5.0


# ===================================================================
#  ExtendedConstraintHandler
# ===================================================================
def test_extended_handler_repair_and_penalty():
    # Encoding: vecsize=2, param "a" of size 1
    enc = DummyParameterExtendingEncoding([("a", 1)])
    # Override vecsize (the Dummy sets it to 1, but we can just set it manually)
    enc.vecsize = 2  # ensure solution part is 2 columns
    # Sub-handlers
    sol_handler = ConstraintHandlerFromLambda(repair_solution_fn=lambda x: x + 1, penalty_fn=lambda x: 0.5)
    param_handler = ConstraintHandlerFromLambda(repair_solution_fn=lambda x: x * 2, penalty_fn=lambda x: 1.0)
    handler = ExtendedConstraintHandler(solution_handler=sol_handler, param_handler_dict={"a": param_handler}, encoding=enc)

    # Full genotype: solution part = [10, 20], param a = [5]
    full = np.array([[10.0, 20.0, 5.0]])

    # Expected repair:
    # decoded solution = [[10,20]] -> repaired = [[11,21]]
    # decoded param = {"a": [[5]]} -> repaired = {"a": [[10]]}
    # encode: base_encoding (DefaultEncoding) identity on solution part -> [[11,21]], then stack params -> [[11,21,10]]
    expected_repaired = np.array([[11.0, 21.0, 10.0]])
    repaired = handler.repair_solution(full)
    assert_array_equal(repaired, expected_repaired)

    # Expected penalty: sol_penalty 0.5 + param_penalty 1.0 = 1.5
    penalty = handler.penalty(full)
    assert penalty == pytest.approx(1.5)
