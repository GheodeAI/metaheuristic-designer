"""
Unit tests for ObjectiveFunc hierarchy.

Contracts verified:
- ObjectiveFromLambda wraps any callable correctly.
- VectorObjectiveFunc exposes lower_bound / upper_bound attributes.
- mode="max" preserves the objective; mode="min" negates it for internal fitness.
- NullObjectiveFunc returns zero fitness for any input.
- __call__ on a Population calls objective for each individual.
"""

import numpy as np
import pytest

from metaheuristic_designer.objective_function import (
    ObjectiveFunc,
    ObjectiveFromLambda,
    NullObjectiveFunc,
)
from metaheuristic_designer.benchmarks.benchmark_funcs import MaxOnes, Sphere
from metaheuristic_designer.population import Population


# ---------------------------------------------------------------------------
# ObjectiveFromLambda basic wiring
# ---------------------------------------------------------------------------

def test_objective_from_lambda_calls_function():
    fn = lambda x: float(x.sum())
    objfunc = ObjectiveFromLambda(fn, mode="max", name="test")
    result = objfunc.objective(np.array([1.0, 2.0, 3.0]))
    assert result == pytest.approx(6.0)


def test_objective_from_lambda_name_stored():
    objfunc = ObjectiveFromLambda(lambda x: 0.0, name="my_func")
    assert objfunc.name == "my_func"


# ---------------------------------------------------------------------------
# mode="max" vs mode="min" fitness polarity
# ---------------------------------------------------------------------------

def test_max_mode_fitness_equals_objective():
    # For a maximisation problem the internal fitness == objective value
    objfunc = MaxOnes(dimension=4)
    geno = np.array([[1, 1, 1, 1]], dtype=float)
    pop = Population(objfunc, geno)
    pop.calculate_fitness()
    assert pop.fitness[0] == pytest.approx(4.0)
    assert pop.objective[0] == pytest.approx(4.0)


def test_min_mode_fitness_is_negated_objective():
    # For a minimisation problem fitness = -objective so that higher fitness is better
    objfunc = Sphere(dimension=2, mode="min")
    geno = np.array([[3.0, 4.0]])   # sphere = 9 + 16 = 25
    pop = Population(objfunc, geno)
    pop.calculate_fitness()
    assert pop.objective[0] == pytest.approx(25.0)
    assert pop.fitness[0] == pytest.approx(-25.0)


# ---------------------------------------------------------------------------
# VectorObjectiveFunc bounds
# ---------------------------------------------------------------------------

def test_vector_objective_func_has_lower_bound():
    objfunc = Sphere(dimension=3, mode="min")
    assert hasattr(objfunc, "lower_bound")
    assert objfunc.lower_bound is not None


def test_vector_objective_func_has_upper_bound():
    objfunc = Sphere(dimension=3, mode="min")
    assert hasattr(objfunc, "upper_bound")
    assert objfunc.upper_bound is not None


def test_vector_objective_func_bounds_exist():
    """Sphere stores lower_bound and upper_bound (can be scalar or array)."""
    objfunc = Sphere(dimension=5, mode="min")
    # bounds can be scalar or 1D array; they must be finite
    lb = objfunc.lower_bound
    ub = objfunc.upper_bound
    assert np.isfinite(lb) if np.isscalar(lb) else np.all(np.isfinite(lb))
    assert np.isfinite(ub) if np.isscalar(ub) else np.all(np.isfinite(ub))
    assert lb < ub if np.isscalar(lb) else np.all(lb < ub)


# ---------------------------------------------------------------------------
# NullObjectiveFunc
# ---------------------------------------------------------------------------

def test_null_objective_returns_zero():
    objfunc = NullObjectiveFunc()
    result = objfunc.objective(np.array([99.0, -3.0, 0.5]))
    assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Population-level fitness evaluation
# ---------------------------------------------------------------------------

def test_population_fitness_all_evaluated(onemax_func):
    geno = np.array([[1, 0, 1, 0, 1, 0, 1, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0]], dtype=float)
    pop = Population(onemax_func, geno)
    pop.calculate_fitness()
    assert pop.fitness[0] == pytest.approx(4.0)
    assert pop.fitness[1] == pytest.approx(0.0)


def test_sphere_known_value():
    # Sphere([0, 0, 0]) should return 0; internal fitness = -0 = 0
    objfunc = Sphere(dimension=3, mode="min")
    geno = np.array([[0.0, 0.0, 0.0]])
    pop = Population(objfunc, geno)
    pop.calculate_fitness()
    assert pop.objective[0] == pytest.approx(0.0)
    assert pop.fitness[0] == pytest.approx(0.0)
