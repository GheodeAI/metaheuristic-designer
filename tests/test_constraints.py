import pytest
from typing import List, Any
import numpy as np
from metaheuristic_designer import ConstraintHandler, NullConstraint
from metaheuristic_designer.constraint_handlers import *

# Test data generators
def generate_test_vectors():
    """Generate various test vectors for parametrization"""
    return [
        np.array([1.0, 2.0, 3.0]),  # Simple case
        np.array([-1.0, 0.0, 1.0]), # With negatives
        np.array([0.0, 0.0, 0.0]),  # All zeros
        np.array([10.0, -10.0, 5.0]), # Mixed positive/negative
        np.array([1.5, 2.5, 3.5]),  # Floats
        np.array([100.0, -100.0, 50.0]), # Large values
        np.array([1e-10, -1e-10, 0.0]), # Very small values
    ]


def generate_bounds():
    """Generate various bound configurations"""
    return [
        (3, -1.0, 1.0),   # Standard bounds
        (3, 0.0, 5.0),    # Positive bounds
        (3, -5.0, 0.0),   # Negative bounds
        (3, -10.0, 10.0), # Wide bounds
        (3, 0.0, 0.0),    # Zero bounds (edge case)
    ]


def generate_different_sizes():
    """Generate different vector sizes for testing"""
    return [
        (1, -1.0, 1.0),   # Single element
        (2, -1.0, 1.0),   # Two elements
        (5, -1.0, 1.0),   # Five elements
        (10, -1.0, 1.0),  # Ten elements
    ]


# NullConstraint Tests
@pytest.mark.parametrize("solution", generate_test_vectors())
def test_null_constraint_repair_returns_identical(solution):
    """Test that NullConstraint repair returns the exact same solution"""
    constraint = NullConstraint()
    repaired = constraint.repair_solution(solution)
    
    np.testing.assert_array_equal(repaired, solution)
    assert repaired is not solution  # Should be a new array


@pytest.mark.parametrize("solution", generate_test_vectors())
def test_null_constraint_penalty_always_zero(solution):
    """Test that NullConstraint always returns 0 penalty"""
    constraint = NullConstraint()
    penalty = constraint.penalty(solution)
    
    assert penalty == 0.0


# LinearPenaltyBoundConstraint Tests
@pytest.mark.parametrize("vecsize,low_lim,up_lim", generate_bounds())
@pytest.mark.parametrize("solution", generate_test_vectors())
def test_linear_penalty_repair_unchanged(vecsize, low_lim, up_lim, solution):
    """Test that LinearPenaltyBoundConstraint doesn't modify solution during repair"""
    constraint = LinearPenaltyBoundConstraint(vecsize=vecsize, low_lim=low_lim, up_lim=up_lim, alpha=1.0)
    repaired = constraint.repair_solution(solution)
    
    np.testing.assert_array_equal(repaired, solution)
    assert repaired is not solution


@pytest.mark.parametrize("vecsize,low_lim,up_lim,alpha,solution,expected_penalty", [
    (3, -1.0, 1.0, 1.0, np.array([0.0, 0.0, 0.0]), 0.0),
    (3, -1.0, 1.0, 1.0, np.array([2.0, 2.0, 2.0]), 3.0),  # All above upper bound
    (3, -1.0, 1.0, 1.0, np.array([-2.0, -2.0, -2.0]), 3.0),  # All below lower bound
    (3, -1.0, 1.0, 1.0, np.array([-2.0, 0.0, 2.0]), 2.0),  # Mixed
    (3, 0.0, 5.0, 2.0, np.array([6.0, 7.0, 8.0]), 12.0),  # Different bounds and alpha
])
def test_linear_penalty_calculation(vecsize, low_lim, up_lim, alpha, solution, expected_penalty):
    """Test linear penalty calculation for various scenarios"""
    constraint = LinearPenaltyBoundConstraint(vecsize=vecsize, low_lim=low_lim, up_lim=up_lim, alpha=alpha)
    penalty = constraint.penalty(solution)
    
    assert penalty == pytest.approx(expected_penalty)


# ClipBoundConstraint Tests
@pytest.mark.parametrize("vecsize,low_lim,up_lim,solution,expected_repaired", [
    (3, -1.0, 1.0, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),  # Within bounds
    (3, -1.0, 1.0, np.array([2.0, 2.0, 2.0]), np.array([1.0, 1.0, 1.0])),  # Above bounds
    (3, -1.0, 1.0, np.array([-2.0, -2.0, -2.0]), np.array([-1.0, -1.0, -1.0])),  # Below bounds
    (3, -1.0, 1.0, np.array([-2.0, 0.0, 2.0]), np.array([-1.0, 0.0, 1.0])),  # Mixed
    (3, 0.0, 5.0, np.array([-1.0, 3.0, 6.0]), np.array([0.0, 3.0, 5.0])),  # Different bounds
])
def test_clip_constraint_repair(vecsize, low_lim, up_lim, solution, expected_repaired):
    """Test that ClipBoundConstraint properly clips values to bounds"""
    constraint = ClipBoundConstraint(vecsize=vecsize, low_lim=low_lim, up_lim=up_lim)
    repaired = constraint.repair_solution(solution)
    
    np.testing.assert_array_equal(repaired, expected_repaired)
    assert repaired is not solution


@pytest.mark.parametrize("vecsize,low_lim,up_lim", generate_bounds())
@pytest.mark.parametrize("solution", generate_test_vectors())
def test_clip_constraint_penalty_zero(vecsize, low_lim, up_lim, solution):
    """Test that ClipBoundConstraint always returns 0 penalty"""
    constraint = ClipBoundConstraint(vecsize=vecsize, low_lim=low_lim, up_lim=up_lim)
    penalty = constraint.penalty(solution)
    
    assert penalty == 0.0


# CycleBoundConstraint Tests
@pytest.mark.parametrize("vecsize,low_lim,up_lim,solution,expected_repaired", [
    # Test cases for bounds (-1.0, 1.0)
    (3, -1.0, 1.0, np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0])),  # Within bounds
    (3, -1.0, 1.0, np.array([1.5, 2.0, 2.5]), np.array([-0.5, 0.0, 0.5])),  # Above bounds, cycle back
    (3, -1.0, 1.0, np.array([-1.5, -2.0, -2.5]), np.array([0.5, 0.0, -0.5])),  # Below bounds, cycle forward
    
    # Test cases for bounds (0.0, 5.0)
    (3, 0.0, 5.0, np.array([2.0, 3.0, 4.0]), np.array([2.0, 3.0, 4.0])),  # Within bounds
    (3, 0.0, 5.0, np.array([6.0, 7.0, 8.0]), np.array([1.0, 2.0, 3.0])),  # Above bounds, cycle back
    (3, 0.0, 5.0, np.array([-1.0, -2.0, -3.0]), np.array([4.0, 3.0, 2.0])),  # Below bounds, cycle forward
])
def test_cycle_constraint_repair(vecsize, low_lim, up_lim, solution, expected_repaired):
    """Test that CycleBoundConstraint properly cycles values within bounds"""
    constraint = CycleBoundConstraint(vecsize=vecsize, low_lim=low_lim, up_lim=up_lim)
    repaired = constraint.repair_solution(solution)
    
    np.testing.assert_array_equal(repaired, expected_repaired)
    assert repaired is not solution


@pytest.mark.parametrize("vecsize,low_lim,up_lim", generate_bounds())
@pytest.mark.parametrize("solution", generate_test_vectors())
def test_cycle_constraint_penalty_zero(vecsize, low_lim, up_lim, solution):
    """Test that CycleBoundConstraint always returns 0 penalty"""
    constraint = CycleBoundConstraint(vecsize=vecsize, low_lim=low_lim, up_lim=up_lim)
    penalty = constraint.penalty(solution)
    
    assert penalty == 0.0


# BounceBoundConstraint Tests
@pytest.mark.parametrize("vecsize,low_lim,up_lim,solution,expected_repaired", [
    # Test cases for bounds (-1.0, 1.0)
    (3, -1.0, 1.0, np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0])),  # Within bounds
    (3, -1.0, 1.0, np.array([1.5, 2.0, 2.5]), np.array([0.5, 0.0, -0.5])),  # Single bounce
    (3, -1.0, 1.0, np.array([3.5, 4.0, 4.5]), np.array([-0.5, 0.0, 0.5])),  # Multiple bounces
    (3, -1.0, 1.0, np.array([-1.5, -2.0, -2.5]), np.array([-0.5, 0.0, 0.5])),  # Negative bounces
    
    # Test cases for bounds (0.0, 5.0)
    (3, 0.0, 5.0, np.array([2.0, 3.0, 4.0]), np.array([2.0, 3.0, 4.0])),  # Within bounds
    (3, 0.0, 5.0, np.array([6.0, 7.0, 8.0]), np.array([4.0, 3.0, 2.0])),  # Single bounce
    (3, 0.0, 5.0, np.array([11.0, 12.0, 13.0]), np.array([1.0, 2.0, 3.0])),  # Multiple bounces
])
def test_bounce_constraint_repair(vecsize, low_lim, up_lim, solution, expected_repaired):
    """Test that BounceBoundConstraint properly bounces values within bounds"""
    constraint = BounceBoundConstraint(vecsize=vecsize, low_lim=low_lim, up_lim=up_lim)
    repaired = constraint.repair_solution(solution)
    
    np.testing.assert_array_equal(repaired, expected_repaired)
    assert repaired is not solution


@pytest.mark.parametrize("vecsize,low_lim,up_lim", generate_bounds())
@pytest.mark.parametrize("solution", generate_test_vectors())
def test_bounce_constraint_penalty_zero(vecsize, low_lim, up_lim, solution):
    """Test that BounceBoundConstraint always returns 0 penalty"""
    constraint = BounceBoundConstraint(vecsize=vecsize, low_lim=low_lim, up_lim=up_lim)
    penalty = constraint.penalty(solution)
    
    assert penalty == 0.0


# CompositeConstraint Tests
def test_composite_constraint_initialization():
    """Test that CompositeConstraint can be initialized with constraint list"""
    constraints = [
        NullConstraint(),
        ClipBoundConstraint(vecsize=3, low_lim=-1.0, up_lim=1.0),
        CycleBoundConstraint(vecsize=3, low_lim=0.0, up_lim=5.0)
    ]
    
    composite = CompositeConstraint(constraints)
    assert len(composite.constraints) == 3


@pytest.mark.parametrize("solution", generate_test_vectors())
def test_composite_constraint_repair_sequence(solution):
    """Test that CompositeConstraint applies constraints in sequence"""
    # Create constraints that modify the solution in predictable ways
    clip_constraint = ClipBoundConstraint(vecsize=3, low_lim=-1.0, up_lim=1.0)
    null_constraint = NullConstraint()
    
    constraints = [clip_constraint, null_constraint]
    composite = CompositeConstraint(constraints)
    
    # Apply individually for comparison
    expected = clip_constraint.repair_solution(solution)
    expected = null_constraint.repair_solution(expected)
    
    repaired = composite.repair_solution(solution)
    
    np.testing.assert_array_equal(repaired, expected)
    assert repaired is not solution


@pytest.mark.parametrize("solution", generate_test_vectors())
def test_composite_constraint_penalty_sum(solution):
    """Test that CompositeConstraint sums penalties from all constraints"""
    # Use LinearPenaltyBoundConstraint since it's the only one with non-zero penalty
    penalty_constraint1 = LinearPenaltyBoundConstraint(vecsize=3, low_lim=-1.0, up_lim=1.0, alpha=1.0)
    penalty_constraint2 = LinearPenaltyBoundConstraint(vecsize=3, low_lim=0.0, up_lim=5.0, alpha=2.0)
    null_constraint = NullConstraint()
    
    constraints = [penalty_constraint1, penalty_constraint2, null_constraint]
    composite = CompositeConstraint(constraints)
    
    expected_penalty = (penalty_constraint1.penalty(solution) + 
                       penalty_constraint2.penalty(solution) + 
                       null_constraint.penalty(solution))
    
    penalty = composite.penalty(solution)
    
    assert penalty == pytest.approx(expected_penalty)


# Edge case tests
def test_empty_composite_constraint():
    """Test CompositeConstraint with empty constraints list"""
    composite = CompositeConstraint([])
    solution = np.array([1.0, 2.0, 3.0])
    
    repaired = composite.repair_solution(solution)
    penalty = composite.penalty(solution)
    
    np.testing.assert_array_equal(repaired, solution)
    assert penalty == 0.0


@pytest.mark.parametrize("constraint_class", [
    ClipBoundConstraint,
    CycleBoundConstraint,
    BounceBoundConstraint,
])
def test_zero_bounds_edge_case(constraint_class):
    """Test constraints with zero-width bounds"""
    vecsize, low_lim, up_lim = 3, 0.0, 0.0
    constraint = constraint_class(vecsize=vecsize, low_lim=low_lim, up_lim=up_lim)
    solution = np.array([1.0, -1.0, 0.0])
    
    repaired = constraint.repair_solution(solution)
    penalty = constraint.penalty(solution)
    
    # All values should be forced to the bound value (0.0)
    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_array_equal(repaired, expected)


@pytest.mark.parametrize("vecsize,low_lim,up_lim", generate_different_sizes())
def test_different_vector_sizes(vecsize, low_lim, up_lim):
    """Test constraints with different vector sizes"""
    solution = np.random.uniform(low_lim - 2, up_lim + 2, size=vecsize)
    
    clip_constraint = ClipBoundConstraint(vecsize=vecsize, low_lim=low_lim, up_lim=up_lim)
    repaired = clip_constraint.repair_solution(solution)
    
    # Check that the repaired solution has the correct size and is within bounds
    assert repaired.shape == (vecsize,)
    assert np.all(repaired >= low_lim)
    assert np.all(repaired <= up_lim)


def test_single_element_vector():
    """Test constraints with single-element vectors"""
    vecsize, low_lim, up_lim = 1, 0.0, 1.0
    solution = np.array([5.0])
    
    clip_constraint = ClipBoundConstraint(vecsize=vecsize, low_lim=low_lim, up_lim=up_lim)
    repaired = clip_constraint.repair_solution(solution)
    
    np.testing.assert_array_equal(repaired, np.array([1.0]))


# Property-based tests
@pytest.mark.parametrize("constraint_class", [
    ClipBoundConstraint,
    CycleBoundConstraint,
    BounceBoundConstraint,
])
@pytest.mark.parametrize("vecsize,low_lim,up_lim", generate_different_sizes())
def test_repair_preserves_shape(constraint_class, vecsize, low_lim, up_lim):
    """Test that repair operations preserve array shape"""
    constraint = constraint_class(vecsize=vecsize, low_lim=low_lim, up_lim=up_lim)
    
    solution = np.random.randn(vecsize)
    repaired = constraint.repair_solution(solution)
    
    assert repaired.shape == solution.shape


@pytest.mark.parametrize("constraint_class", [
    ClipBoundConstraint,
    CycleBoundConstraint,
    BounceBoundConstraint,
])
@pytest.mark.parametrize("vecsize,low_lim,up_lim", generate_bounds())
def test_repair_respects_bounds(constraint_class, vecsize, low_lim, up_lim):
    """Test that repaired solutions respect the specified bounds"""
    constraint = constraint_class(vecsize=vecsize, low_lim=low_lim, up_lim=up_lim)
    
    # Test with random solutions
    for _ in range(10):
        solution = np.random.uniform(low_lim - 5, up_lim + 5, size=vecsize)
        repaired = constraint.repair_solution(solution)
        
        # All repaired values should be within bounds
        assert np.all(repaired >= low_lim)
        assert np.all(repaired <= up_lim)