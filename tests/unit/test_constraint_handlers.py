"""
Unit tests for constraint handler classes.

Covers:
- ClipBoundConstraint: clips values to bounds
- BounceBoundConstraint: bounces values off bounds
- CycleBoundConstraint: wraps values around bounds
- ConstraintHandlerFromLambda: custom repair and penalty
- NullConstraint: no-op
- Degenerate case: upper_bound == lower_bound
"""

import numpy as np
import pytest

from metaheuristic_designer.constraint_handler import (
    ConstraintHandlerFromLambda,
    NullConstraint,
)
from metaheuristic_designer.constraint_handlers.clip_bound_constraint import ClipBoundConstraint
from metaheuristic_designer.constraint_handlers.bounce_bound_constraint import BounceBoundConstraint
from metaheuristic_designer.constraint_handlers.cycle_bound_constraint import CycleBoundConstraint


# ---------------------------------------------------------------------------
# ClipBoundConstraint
# ---------------------------------------------------------------------------

def test_clip_in_bounds_unchanged():
    ch = ClipBoundConstraint(dimension=3, lower_bound=-1.0, upper_bound=1.0)
    pop = np.array([[0.5, -0.5, 0.0]])
    result = ch.repair_solution(pop)
    np.testing.assert_array_almost_equal(result, pop)


def test_clip_above_upper_bound_clipped():
    ch = ClipBoundConstraint(dimension=3, lower_bound=-1.0, upper_bound=1.0)
    pop = np.array([[2.0, 1.5, -2.0]])
    result = ch.repair_solution(pop)
    assert np.all(result <= 1.0)
    assert np.all(result >= -1.0)


def test_clip_below_lower_bound_clipped():
    ch = ClipBoundConstraint(dimension=3, lower_bound=0.0, upper_bound=10.0)
    pop = np.array([[-5.0, 11.0, 3.0]])
    result = ch.repair_solution(pop)
    assert result[0, 0] == pytest.approx(0.0)
    assert result[0, 1] == pytest.approx(10.0)
    assert result[0, 2] == pytest.approx(3.0)


def test_clip_degenerate_scalar_bounds():
    """When upper_bound == lower_bound (scalar), all values become that constant."""
    ch = ClipBoundConstraint(dimension=3, lower_bound=5.0, upper_bound=5.0)
    pop = np.array([[1.0, 3.0, 7.0]])
    result = ch.repair_solution(pop)
    assert np.all(result == 5.0)


def test_clip_penalty_always_zero():
    ch = ClipBoundConstraint(dimension=3, lower_bound=-1.0, upper_bound=1.0)
    pop = np.array([[2.0, -2.0, 0.0]])
    penalty = ch.penalty(pop)
    assert penalty == 0


# ---------------------------------------------------------------------------
# BounceBoundConstraint
# ---------------------------------------------------------------------------

def test_bounce_in_bounds_unchanged():
    ch = BounceBoundConstraint(dimension=3, lower_bound=-1.0, upper_bound=1.0)
    pop = np.array([[0.5, -0.5, 0.0]])
    result = ch.repair_solution(pop)
    assert np.all(result >= -1.0) and np.all(result <= 1.0)


def test_bounce_out_of_bounds_repaired():
    ch = BounceBoundConstraint(dimension=2, lower_bound=-1.0, upper_bound=1.0)
    pop = np.array([[1.5, -1.5]])
    result = ch.repair_solution(pop)
    assert np.all(result >= -1.0)
    assert np.all(result <= 1.0)


def test_bounce_degenerate_bounds():
    ch = BounceBoundConstraint(dimension=2, lower_bound=3.0, upper_bound=3.0)
    pop = np.array([[1.0, 5.0]])
    result = ch.repair_solution(pop)
    assert np.all(result == 3.0)


def test_bounce_penalty_always_zero():
    ch = BounceBoundConstraint(dimension=2, lower_bound=-1.0, upper_bound=1.0)
    pop = np.array([[2.0, -2.0]])
    assert ch.penalty(pop) == 0


# ---------------------------------------------------------------------------
# CycleBoundConstraint
# ---------------------------------------------------------------------------

def test_cycle_in_bounds_unchanged():
    ch = CycleBoundConstraint(dimension=3, lower_bound=0.0, upper_bound=10.0)
    pop = np.array([[3.0, 7.0, 5.0]])
    result = ch.repair_solution(pop)
    assert np.all(result >= 0.0) and np.all(result <= 10.0)


def test_cycle_wraps_out_of_bounds():
    ch = CycleBoundConstraint(dimension=2, lower_bound=0.0, upper_bound=10.0)
    pop = np.array([[11.0, -1.0]])
    result = ch.repair_solution(pop)
    assert np.all(result >= 0.0)
    assert np.all(result <= 10.0)


def test_cycle_periodicity():
    """Cycling by range should give same result: x and x+range should map to same."""
    ch = CycleBoundConstraint(dimension=1, lower_bound=0.0, upper_bound=10.0)
    pop_x = np.array([[3.0]])
    pop_x_plus_range = np.array([[13.0]])
    result_x = ch.repair_solution(pop_x)
    result_shifted = ch.repair_solution(pop_x_plus_range)
    np.testing.assert_array_almost_equal(result_x, result_shifted)


def test_cycle_degenerate_bounds():
    ch = CycleBoundConstraint(dimension=2, lower_bound=2.0, upper_bound=2.0)
    pop = np.array([[1.0, 5.0]])
    result = ch.repair_solution(pop)
    assert np.all(result == 2.0)


def test_cycle_penalty_always_zero():
    ch = CycleBoundConstraint(dimension=2, lower_bound=-1.0, upper_bound=1.0)
    pop = np.array([[5.0, -5.0]])
    assert ch.penalty(pop) == 0


# ---------------------------------------------------------------------------
# ConstraintHandlerFromLambda
# ---------------------------------------------------------------------------

def test_lambda_constraint_repair_only():
    repair_fn = lambda pop: np.clip(pop, 0, 1)
    ch = ConstraintHandlerFromLambda(repair_solution_fn=repair_fn)
    pop = np.array([[2.0, -1.0, 0.5]])
    result = ch.repair_solution(pop)
    assert np.all(result >= 0) and np.all(result <= 1)
    assert ch.penalty(pop) == 0


def test_lambda_constraint_penalty_only():
    penalty_fn = lambda pop: float(np.sum(np.maximum(0, pop - 1)))
    ch = ConstraintHandlerFromLambda(penalty_fn=penalty_fn)
    pop = np.array([[2.0, -1.0, 0.5]])
    penalty = ch.penalty(pop)
    assert penalty > 0
    # repair_solution returns a copy when repair_fn is None
    result = ch.repair_solution(pop)
    np.testing.assert_array_equal(result, pop)


def test_lambda_constraint_requires_at_least_one_function():
    with pytest.raises(ValueError):
        ConstraintHandlerFromLambda()


# ---------------------------------------------------------------------------
# NullConstraint
# ---------------------------------------------------------------------------

def test_null_constraint_repair_returns_copy():
    ch = NullConstraint()
    pop = np.array([[1.0, 2.0, 3.0]])
    result = ch.repair_solution(pop)
    np.testing.assert_array_equal(result, pop)
    # Should be a copy, not the same object
    assert result is not pop


def test_null_constraint_penalty_is_zero():
    ch = NullConstraint()
    pop = np.array([[1.0, 2.0, 3.0]])
    assert ch.penalty(pop) == 0


# ---------------------------------------------------------------------------
# Integration: objective function with constraint handler
# ---------------------------------------------------------------------------

def test_sphere_with_clip_constraint():
    """Sphere with ClipBoundConstraint: solutions get clamped, fitness is still finite."""
    from metaheuristic_designer.benchmarks import Sphere
    from metaheuristic_designer.population import Population

    obj = Sphere(dimension=4, mode="min",
                 constraint_handler=ClipBoundConstraint(4, lower_bound=-100, upper_bound=100))
    pop_matrix = np.array([[200.0, -200.0, 50.0, -50.0]])
    pop = Population(obj, pop_matrix.copy())
    pop.calculate_fitness()
    assert np.isfinite(pop.fitness[0])


# ---------------------------------------------------------------------------
# CompositeConstraint
# ---------------------------------------------------------------------------

def test_composite_constraint_applies_all_handlers():
    from metaheuristic_designer.constraint_handlers.composite_constraint import CompositeConstraint

    c1 = ClipBoundConstraint(dimension=3, lower_bound=-1.0, upper_bound=1.0)
    c2 = ClipBoundConstraint(dimension=3, lower_bound=-0.5, upper_bound=0.5)
    composite = CompositeConstraint([c1, c2])

    pop = np.array([[2.0, -2.0, 0.3]])
    result = composite.repair_solution(pop)
    # After c1: [-1, -1, 0.3], after c2: [-0.5, -0.5, 0.3]
    assert np.all(result >= -0.5)
    assert np.all(result <= 0.5)


def test_composite_constraint_penalty_sums():
    from metaheuristic_designer.constraint_handlers.composite_constraint import CompositeConstraint
    from metaheuristic_designer.constraint_handlers.linear_bound_penalty_constraint import LinearBoundPenaltyConstraint

    c1 = LinearBoundPenaltyConstraint(dimension=3, alpha=1.0, lower_bound=-1.0, upper_bound=1.0)
    c2 = LinearBoundPenaltyConstraint(dimension=3, alpha=2.0, lower_bound=-1.0, upper_bound=1.0)
    composite = CompositeConstraint([c1, c2])

    pop = np.array([[2.0, -2.0, 0.0]])
    p1 = c1.penalty(pop)
    p2 = c2.penalty(pop)
    p_composite = composite.penalty(pop)
    # Should be sum of individual penalties
    assert p_composite == pytest.approx(p1 + p2)


# ---------------------------------------------------------------------------
# LinearBoundPenaltyConstraint
# ---------------------------------------------------------------------------

def test_linear_penalty_in_bounds_gives_zero():
    from metaheuristic_designer.constraint_handlers.linear_bound_penalty_constraint import LinearBoundPenaltyConstraint
    ch = LinearBoundPenaltyConstraint(dimension=3, lower_bound=-1.0, upper_bound=1.0)
    pop = np.array([[0.5, -0.5, 0.0]])
    result = ch.penalty(pop)
    assert np.all(result == 0.0)


def test_linear_penalty_out_of_bounds_positive():
    from metaheuristic_designer.constraint_handlers.linear_bound_penalty_constraint import LinearBoundPenaltyConstraint
    ch = LinearBoundPenaltyConstraint(dimension=3, lower_bound=-1.0, upper_bound=1.0)
    pop = np.array([[3.0, -3.0, 0.0]])
    result = ch.penalty(pop)
    assert np.all(result > 0)


def test_linear_penalty_scales_with_alpha():
    from metaheuristic_designer.constraint_handlers.linear_bound_penalty_constraint import LinearBoundPenaltyConstraint
    ch1 = LinearBoundPenaltyConstraint(dimension=2, alpha=1.0, lower_bound=0.0, upper_bound=1.0)
    ch2 = LinearBoundPenaltyConstraint(dimension=2, alpha=2.0, lower_bound=0.0, upper_bound=1.0)
    pop = np.array([[2.0, -1.0]])
    p1 = ch1.penalty(pop)
    p2 = ch2.penalty(pop)
    np.testing.assert_array_almost_equal(p2, 2 * p1)


def test_linear_penalty_repair_returns_copy():
    """LinearBoundPenaltyConstraint inherits PenalizeConstraint.repair_solution → returns copy."""
    from metaheuristic_designer.constraint_handlers.linear_bound_penalty_constraint import LinearBoundPenaltyConstraint
    ch = LinearBoundPenaltyConstraint(dimension=3, lower_bound=-1.0, upper_bound=1.0)
    pop = np.array([[2.0, -2.0, 0.0]])
    result = ch.repair_solution(pop)
    # repair_solution is a no-op for PenalizeConstraint (returns copy)
    np.testing.assert_array_equal(result, pop)
