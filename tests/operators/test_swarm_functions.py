import numpy as np
from numpy.testing import assert_array_equal

from conftest import rng, pso_population

from metaheuristic_designer.operators.operator_functions.swarm import (
    pso_operator,
    firefly_operator,
)


# ===================================================================
#  pso_operator (raw array function)
# ===================================================================
def test_pso_operator_deterministic(rng):
    pop = np.array([[1.0, 2.0], [3.0, 4.0]])
    speed = np.array([[0.1, 0.1], [0.2, 0.2]])
    hist_best = pop.copy()
    global_best = np.array([3.0, 4.0])

    new_pop, new_speed = pso_operator(pop, speed, hist_best, global_best, random_state=rng, w=0.7, c1=1.5, c2=1.5)
    assert new_pop.shape == pop.shape
    assert new_speed.shape == speed.shape

    # Reproducibility
    rng2 = np.random.default_rng(42)
    exp_pop, exp_speed = pso_operator(pop.copy(), speed.copy(), hist_best.copy(), global_best.copy(),
                                      random_state=rng2, w=0.7, c1=1.5, c2=1.5)
    assert_array_equal(new_pop, exp_pop)
    assert_array_equal(new_speed, exp_speed)


# ===================================================================
#  firefly_operator
# ===================================================================
def test_firefly_operator_deterministic(rng):
    pop = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    fit = np.array([0.1, 0.9, 0.3])
    result = firefly_operator(pop.copy(), fit, random_state=rng, alpha_0=0.2, beta_0=1.0, delta=1.0, gamma=1.0)
    assert result.shape == pop.shape

    rng2 = np.random.default_rng(42)
    expected = firefly_operator(pop.copy(), fit.copy(), random_state=rng2, alpha_0=0.2, beta_0=1.0, delta=1.0, gamma=1.0)
    assert_array_equal(result, expected)


def test_firefly_operator_no_change_when_only_one_individual(rng):
    pop = np.array([[1.0, 2.0]])
    fit = np.array([0.5])
    result = firefly_operator(pop.copy(), fit, random_state=rng)
    assert_array_equal(result, pop)