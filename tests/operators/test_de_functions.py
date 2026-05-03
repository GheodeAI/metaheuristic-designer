import pytest
import numpy as np
from numpy.testing import assert_array_equal

from conftest import rng, de_pop, de_fitness

from metaheuristic_designer.operators.operator_functions.differential_evolution import (
    differential_evolution_rand1,
    differential_evolution_best1,
    differential_evolution_rand2,
    differential_evolution_best2,
    differential_evolution_current_to_rand1,
    differential_evolution_current_to_best1,
    differential_evolution_current_to_pbest1,
)


# ===================================================================
#  Minimum size validation
# ===================================================================
def test_de_rand1_raises_on_too_small():
    with pytest.raises(ValueError):
        differential_evolution_rand1(np.zeros((3, 2)), None)


def test_de_best1_raises_on_too_small():
    with pytest.raises(ValueError):
        differential_evolution_best1(np.zeros((2, 2)), None)


def test_de_rand2_raises_on_too_small():
    with pytest.raises(ValueError):
        differential_evolution_rand2(np.zeros((5, 2)), None)


def test_de_best2_raises_on_too_small():
    with pytest.raises(ValueError):
        differential_evolution_best2(np.zeros((4, 2)), None)


def test_de_current_to_rand1_raises_on_too_small():
    with pytest.raises(ValueError):
        differential_evolution_current_to_rand1(np.zeros((3, 2)), None)


def test_de_current_to_best1_raises_on_too_small():
    with pytest.raises(ValueError):
        differential_evolution_current_to_best1(np.zeros((2, 2)), None)


def test_de_current_to_pbest1_raises_on_too_small():
    with pytest.raises(ValueError):
        differential_evolution_current_to_pbest1(np.zeros((2, 2)), None)


# ===================================================================
#  DE/rand/1 – works with de_pop (4)
# ===================================================================
def test_de_rand1_value(rng, de_pop):
    result = differential_evolution_rand1(de_pop.copy(), None, random_state=rng, F=0.8, Cr=0.9)
    rng2 = np.random.default_rng(42)
    expected = differential_evolution_rand1(de_pop.copy(), None, random_state=rng2, F=0.8, Cr=0.9)
    assert_array_equal(result, expected)


def test_de_rand1_preserves_shape(rng, de_pop):
    result = differential_evolution_rand1(de_pop.copy(), None, random_state=rng)
    assert result.shape == de_pop.shape


# ===================================================================
#  DE/best/1 – works with de_pop (4)
# ===================================================================
def test_de_best1_value(rng, de_pop, de_fitness):
    result = differential_evolution_best1(de_pop.copy(), de_fitness, random_state=rng, F=0.8, Cr=0.9)
    rng2 = np.random.default_rng(42)
    expected = differential_evolution_best1(de_pop.copy(), de_fitness, random_state=rng2, F=0.8, Cr=0.9)
    assert_array_equal(result, expected)


# ===================================================================
#  DE/rand/2 – needs 6 individuals
# ===================================================================
def test_de_rand2_value(rng):
    pop = np.random.default_rng(42).uniform(0, 1, (6, 2))
    result = differential_evolution_rand2(pop.copy(), None, random_state=rng, F=0.8, Cr=0.9)
    rng2 = np.random.default_rng(42)
    expected = differential_evolution_rand2(pop.copy(), None, random_state=rng2, F=0.8, Cr=0.9)
    assert_array_equal(result, expected)


# ===================================================================
#  DE/best/2 – needs 6 individuals
# ===================================================================
def test_de_best2_value(rng):
    pop = np.random.default_rng(42).uniform(0, 1, (6, 2))
    fit = np.random.default_rng(42).uniform(0, 1, 6)
    result = differential_evolution_best2(pop.copy(), fit, random_state=rng, F=0.8, Cr=0.9)
    rng2 = np.random.default_rng(42)
    expected = differential_evolution_best2(pop.copy(), fit, random_state=rng2, F=0.8, Cr=0.9)
    assert_array_equal(result, expected)


# ===================================================================
#  DE/current-to-rand/1 – works with de_pop (4)
# ===================================================================
def test_de_current_to_rand1_value(rng, de_pop):
    result = differential_evolution_current_to_rand1(de_pop.copy(), None, random_state=rng, F=0.8, Cr=0.9)
    rng2 = np.random.default_rng(42)
    expected = differential_evolution_current_to_rand1(de_pop.copy(), None, random_state=rng2, F=0.8, Cr=0.9)
    assert_array_equal(result, expected)


# ===================================================================
#  DE/current-to-best/1 – works with de_pop (4)
# ===================================================================
def test_de_current_to_best1_value(rng, de_pop, de_fitness):
    result = differential_evolution_current_to_best1(de_pop.copy(), de_fitness, random_state=rng, F=0.8, Cr=0.9)
    rng2 = np.random.default_rng(42)
    expected = differential_evolution_current_to_best1(de_pop.copy(), de_fitness, random_state=rng2, F=0.8, Cr=0.9)
    assert_array_equal(result, expected)


# ===================================================================
#  DE/current-to-pbest/1 – works with de_pop (4)
# ===================================================================
def test_de_current_to_pbest1_value(rng, de_pop, de_fitness):
    result = differential_evolution_current_to_pbest1(de_pop.copy(), de_fitness, random_state=rng, p=0.5, F=0.8, Cr=0.9)
    rng2 = np.random.default_rng(42)
    expected = differential_evolution_current_to_pbest1(de_pop.copy(), de_fitness, random_state=rng2, p=0.5, F=0.8, Cr=0.9)
    assert_array_equal(result, expected)
