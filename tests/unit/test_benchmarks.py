"""
Unit tests for benchmark objective functions.

Contract:
- All benchmark functions must return finite values for valid inputs.
- Mode "min" means fitness = -objective; mode "max" means fitness = objective.
- Sphere at the origin returns 0 (global minimum).
- MaxOnes at all-ones returns dimension (maximum).
- Bounds are accessible and valid.
"""

import numpy as np
import pytest

from metaheuristic_designer.benchmarks.benchmark_funcs import (
    MaxOnes,
    Sphere,
    Rosenbrock,
    Ackley,
    Rastrigin,
    Griewank,
    BentCigar,
    Discus,
    HighCondElliptic,
    Weierstrass,
    ModSchwefel,
    Katsuura,
    HappyCat,
    HGBat,
    ExpandedGriewankPlusRosenbrock,
    ExpandedShafferF6,
    SumPowell,
    N4XinSheYang,
    DiophantineEq,
)
from metaheuristic_designer.population import Population


DIM = 4
ZEROS = np.zeros((1, DIM))
ONES = np.ones((1, DIM))
RANDOM = np.array([[1.5, -0.3, 2.1, -1.0]])


def _eval(objfunc, pop_matrix):
    """Create a population and evaluate it; return objective values."""
    pop = Population(objfunc, pop_matrix.copy())
    pop.calculate_fitness()
    return pop.objective


# ---------------------------------------------------------------------------
# MaxOnes
# ---------------------------------------------------------------------------

def test_maxones_allones_returns_dimension():
    obj = MaxOnes(dimension=DIM)
    result = _eval(obj, ONES)
    assert result[0] == pytest.approx(DIM)


def test_maxones_allzeros_returns_zero():
    obj = MaxOnes(dimension=DIM)
    result = _eval(obj, ZEROS)
    assert result[0] == pytest.approx(0.0)


def test_maxones_mode_max_fitness_equals_objective():
    obj = MaxOnes(dimension=DIM, mode="max")
    pop = Population(obj, ONES.copy())
    pop.calculate_fitness()
    assert pop.fitness[0] == pytest.approx(pop.objective[0])


def test_maxones_bounds():
    obj = MaxOnes(dimension=DIM)
    assert obj.lower_bound == 0
    assert obj.upper_bound == 1


# ---------------------------------------------------------------------------
# Sphere
# ---------------------------------------------------------------------------

def test_sphere_at_origin_returns_zero():
    obj = Sphere(dimension=DIM, mode="min")
    result = _eval(obj, ZEROS)
    assert result[0] == pytest.approx(0.0, abs=1e-12)


def test_sphere_returns_finite_for_random_input():
    obj = Sphere(dimension=DIM, mode="min")
    result = _eval(obj, RANDOM)
    assert np.isfinite(result[0])


def test_sphere_min_mode_fitness_negates_objective():
    obj = Sphere(dimension=DIM, mode="min")
    pop = Population(obj, RANDOM.copy())
    pop.calculate_fitness()
    assert pop.fitness[0] == pytest.approx(-pop.objective[0])


def test_sphere_dimension_stored():
    obj = Sphere(dimension=5, mode="min")
    assert obj.dimension == 5


# ---------------------------------------------------------------------------
# Rosenbrock
# ---------------------------------------------------------------------------

def test_rosenbrock_finite_at_random():
    obj = Rosenbrock(dimension=DIM, mode="min")
    result = _eval(obj, RANDOM)
    assert np.isfinite(result[0])


def test_rosenbrock_at_ones_near_zero():
    """Rosenbrock global minimum is 0 at (1, 1, ..., 1)."""
    obj = Rosenbrock(dimension=DIM, mode="min")
    result = _eval(obj, ONES)
    assert result[0] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Ackley
# ---------------------------------------------------------------------------

def test_ackley_finite_at_random():
    obj = Ackley(dimension=DIM, mode="min")
    result = _eval(obj, RANDOM)
    assert np.isfinite(result[0])


def test_ackley_at_origin_near_zero():
    obj = Ackley(dimension=DIM, mode="min")
    result = _eval(obj, ZEROS)
    assert result[0] == pytest.approx(0.0, abs=1e-8)


# ---------------------------------------------------------------------------
# Rastrigin
# ---------------------------------------------------------------------------

def test_rastrigin_finite_at_random():
    obj = Rastrigin(dimension=DIM, mode="min")
    result = _eval(obj, RANDOM)
    assert np.isfinite(result[0])


def test_rastrigin_at_origin_near_zero():
    obj = Rastrigin(dimension=DIM, mode="min")
    result = _eval(obj, ZEROS)
    assert result[0] == pytest.approx(0.0, abs=1e-8)


# ---------------------------------------------------------------------------
# Griewank
# ---------------------------------------------------------------------------

def test_griewank_finite_at_random():
    obj = Griewank(dimension=DIM, mode="min")
    result = _eval(obj, RANDOM)
    assert np.isfinite(result[0])


# ---------------------------------------------------------------------------
# BentCigar
# ---------------------------------------------------------------------------

def test_bentcigar_finite_at_random():
    obj = BentCigar(dimension=DIM, mode="min")
    result = _eval(obj, RANDOM)
    assert np.isfinite(result[0])


def test_bentcigar_at_origin_near_zero():
    obj = BentCigar(dimension=DIM, mode="min")
    result = _eval(obj, ZEROS)
    assert result[0] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Discus
# ---------------------------------------------------------------------------

def test_discus_finite_at_random():
    obj = Discus(dimension=DIM, mode="min")
    result = _eval(obj, RANDOM)
    assert np.isfinite(result[0])


# ---------------------------------------------------------------------------
# HighCondElliptic
# ---------------------------------------------------------------------------

def test_high_cond_elliptic_finite():
    obj = HighCondElliptic(dimension=DIM, mode="min")
    result = _eval(obj, RANDOM)
    assert np.isfinite(result[0])


# ---------------------------------------------------------------------------
# Weierstrass
# ---------------------------------------------------------------------------

def test_weierstrass_finite():
    obj = Weierstrass(dimension=DIM, mode="min")
    result = _eval(obj, RANDOM)
    assert np.isfinite(result[0])


# ---------------------------------------------------------------------------
# ModSchwefel
# ---------------------------------------------------------------------------

def test_mod_schwefel_finite():
    obj = ModSchwefel(dimension=DIM, mode="min")
    result = _eval(obj, RANDOM)
    assert np.isfinite(result[0])


# ---------------------------------------------------------------------------
# Katsuura
# ---------------------------------------------------------------------------

def test_katsuura_finite():
    obj = Katsuura(dimension=DIM, mode="min")
    result = _eval(obj, RANDOM)
    assert np.isfinite(result[0])


# ---------------------------------------------------------------------------
# HappyCat
# ---------------------------------------------------------------------------

def test_happycat_finite():
    obj = HappyCat(dimension=DIM, mode="min")
    input_data = np.array([[0.5, -0.5, 1.0, -1.0]])
    result = _eval(obj, input_data)
    assert np.isfinite(result[0])


# ---------------------------------------------------------------------------
# HGBat
# ---------------------------------------------------------------------------

def test_hgbat_finite():
    obj = HGBat(dimension=DIM, mode="min")
    input_data = np.array([[0.5, -0.5, 1.0, -1.0]])
    result = _eval(obj, input_data)
    assert np.isfinite(result[0])


# ---------------------------------------------------------------------------
# ExpandedGriewankPlusRosenbrock
# ---------------------------------------------------------------------------

def test_expanded_griewank_plus_rosenbrock_finite():
    obj = ExpandedGriewankPlusRosenbrock(dimension=DIM, mode="min")
    result = _eval(obj, RANDOM)
    assert np.isfinite(result[0])


# ---------------------------------------------------------------------------
# ExpandedShafferF6
# ---------------------------------------------------------------------------

def test_expanded_shaffer_f6_finite():
    obj = ExpandedShafferF6(dimension=DIM, mode="min")
    result = _eval(obj, RANDOM)
    assert np.isfinite(result[0])


# ---------------------------------------------------------------------------
# SumPowell
# ---------------------------------------------------------------------------

def test_sum_powell_finite():
    obj = SumPowell(dimension=DIM, mode="min")
    input_data = np.array([[0.1, -0.1, 0.2, -0.2]])
    result = _eval(obj, input_data)
    assert np.isfinite(result[0])


# ---------------------------------------------------------------------------
# N4XinSheYang
# ---------------------------------------------------------------------------

def test_n4_xin_she_yang_finite():
    obj = N4XinSheYang(dimension=DIM, mode="min")
    input_data = np.array([[1.0, -1.0, 2.0, -2.0]])
    result = _eval(obj, input_data)
    assert np.isfinite(result[0])


# ---------------------------------------------------------------------------
# DiophantineEq (non-vectorized objective function)
# ---------------------------------------------------------------------------

def test_diophantine_eq_finite():
    obj = DiophantineEq(dimension=3, coeff=np.array([1.0, 2.0, 3.0]), target=6.0)
    # [1, 1, 1] gives sum = 1 + 2 + 3 = 6 = target → objective = 0
    sol = np.array([1.0, 1.0, 1.0])
    result = obj.objective(sol)
    assert result == pytest.approx(0.0)


def test_diophantine_eq_nonzero():
    obj = DiophantineEq(dimension=3, coeff=np.array([1.0, 2.0, 3.0]), target=10.0)
    sol = np.array([1.0, 1.0, 1.0])
    result = obj.objective(sol)
    assert result == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Multiple-individual population evaluation
# ---------------------------------------------------------------------------

def test_sphere_multiple_individuals_all_finite():
    obj = Sphere(dimension=DIM, mode="min")
    rng = np.random.default_rng(42)
    pop_matrix = rng.uniform(-100, 100, size=(10, DIM))
    pop = Population(obj, pop_matrix)
    pop.calculate_fitness()
    assert np.all(np.isfinite(pop.objective))
    assert np.all(np.isfinite(pop.fitness))
