import pytest
import numpy as np
from metaheuristic_designer.selectionMethods.parent_selection_functions import * 
import metaheuristic_designer as mhd

mhd.reset_seed(0)

example_fitness = np.array([-10, -2, -1, 0, 0, 1, 2, 10])

@pytest.mark.parametrize("fitness", [example_fitness])
@pytest.mark.parametrize("amount, expected", [
    (1, np.array([7])),
    (2, np.array([7,6])),
    (5, np.array([7,6,5,4,3])),
    (8, np.array([7,6,5,4,3,2,1,0])),
])
def test_select_best(fitness, amount, expected):
    result = select_best(fitness, amount)
    assert result.shape[0] == amount
    np.testing.assert_array_equal(result, expected)

@pytest.mark.parametrize("fitness", [example_fitness])
@pytest.mark.parametrize("amount", [1, 2, 5, 8])
@pytest.mark.parametrize("p", [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
def test_prob_tournament(fitness, amount, p):
    result = prob_tournament(fitness, amount, p)
    assert result.shape[0] == fitness.shape[0]

@pytest.mark.parametrize("fitness", [example_fitness])
@pytest.mark.parametrize("amount", [1, 2, 5, 8])
def test_uniform_selection(fitness, amount):
    result = uniform_selection(fitness, amount)
    assert result.shape[0] == amount

@pytest.mark.parametrize("fitness", [example_fitness])
@pytest.mark.parametrize("amount", [1, 2, 5, 8])
@pytest.mark.parametrize("method", [SelectionDist.FIT_PROP, SelectionDist.EXP_RANK, SelectionDist.LIN_RANK, SelectionDist.SIGMA_SCALE])
@pytest.mark.parametrize("f", [0, 0.5, 1, 2, 10])
def test_uniform_selection(fitness, amount, method, f):
    result = roulette(fitness, amount, method=method, f=f)
    assert result.shape[0] == amount

@pytest.mark.parametrize("fitness", [example_fitness])
@pytest.mark.parametrize("amount", [1, 2, 5, 8])
@pytest.mark.parametrize("method", [SelectionDist.FIT_PROP, SelectionDist.EXP_RANK, SelectionDist.LIN_RANK, SelectionDist.SIGMA_SCALE])
@pytest.mark.parametrize("f", [0, 0.5, 1, 2, 10])
def test_uniform_selection(fitness, amount, method, f):
    result = sus(fitness, amount, method=method, f=f)
    assert result.shape[0] == amount