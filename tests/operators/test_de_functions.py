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
#  DE/rand/1
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
#  DE/best/1
# ===================================================================
def test_de_best1_value(rng, de_pop, de_fitness):
    result = differential_evolution_best1(de_pop.copy(), de_fitness, random_state=rng, F=0.8, Cr=0.9)

    rng2 = np.random.default_rng(42)
    expected = differential_evolution_best1(de_pop.copy(), de_fitness, random_state=rng2, F=0.8, Cr=0.9)
    assert_array_equal(result, expected)


# ===================================================================
#  DE/rand/2
# ===================================================================
def test_de_rand2_value(rng, de_pop):
    result = differential_evolution_rand2(de_pop.copy(), None, random_state=rng, F=0.8, Cr=0.9)

    rng2 = np.random.default_rng(42)
    expected = differential_evolution_rand2(de_pop.copy(), None, random_state=rng2, F=0.8, Cr=0.9)
    assert_array_equal(result, expected)


# ===================================================================
#  DE/best/2
# ===================================================================
def test_de_best2_value(rng, de_pop, de_fitness):
    result = differential_evolution_best2(de_pop.copy(), de_fitness, random_state=rng, F=0.8, Cr=0.9)

    rng2 = np.random.default_rng(42)
    expected = differential_evolution_best2(de_pop.copy(), de_fitness, random_state=rng2, F=0.8, Cr=0.9)
    assert_array_equal(result, expected)


# ===================================================================
#  DE/current-to-rand/1
# ===================================================================
def test_de_current_to_rand1_value(rng, de_pop):
    result = differential_evolution_current_to_rand1(de_pop.copy(), None, random_state=rng, F=0.8, Cr=0.9)

    rng2 = np.random.default_rng(42)
    expected = differential_evolution_current_to_rand1(de_pop.copy(), None, random_state=rng2, F=0.8, Cr=0.9)
    assert_array_equal(result, expected)


# ===================================================================
#  DE/current-to-best/1
# ===================================================================
def test_de_current_to_best1_value(rng, de_pop, de_fitness):
    result = differential_evolution_current_to_best1(de_pop.copy(), de_fitness, random_state=rng, F=0.8, Cr=0.9)

    rng2 = np.random.default_rng(42)
    expected = differential_evolution_current_to_best1(de_pop.copy(), de_fitness, random_state=rng2, F=0.8, Cr=0.9)
    assert_array_equal(result, expected)


# ===================================================================
#  DE/current-to-pbest/1
# ===================================================================
def test_de_current_to_pbest1_value(rng, de_pop, de_fitness):
    result = differential_evolution_current_to_pbest1(de_pop.copy(), de_fitness, random_state=rng, p=0.5, F=0.8, Cr=0.9)

    rng2 = np.random.default_rng(42)
    expected = differential_evolution_current_to_pbest1(de_pop.copy(), de_fitness, random_state=rng2, p=0.5, F=0.8, Cr=0.9)
    assert_array_equal(result, expected)