import pytest
import numpy as np
from metaheuristic_designer.selectionMethods.survivor_selection_functions import * 
import metaheuristic_designer as mhd

mhd.reset_seed(0)

example_fitness = np.array([-10, -2, -1, 0, 0, 1, 2, 10])
offspring_fitness_better = np.array([-9, 10, 34, 2, 100, 2, 10, 100]) 
offspring_fitness_worse = np.array([-20, -5, -2, -1, -10, -90, -100, -10.1])
offspring_fitness_equal = example_fitness.copy()
offspring_fitness_mixed = np.array([-9, -5, 34, -1, 100, 2, 100, -10.1])

@pytest.mark.parametrize("parent_fitness", [example_fitness])
@pytest.mark.parametrize("offspring_fitness, expected", [
    (offspring_fitness_better, np.array([8,9,10,11,12,13,14,15])),
    (offspring_fitness_worse, np.array([0,1,2,3,4,5,6,7])),
    (offspring_fitness_equal, np.array([8,9,10,11,12,13,14,15])),
    (offspring_fitness_mixed, np.array([8,1,10,3,12,13,14,7]))
])
def test_one_to_one(parent_fitness, offspring_fitness, expected):
    result = one_to_one(parent_fitness, offspring_fitness)
    assert result.max() < len(parent_fitness) + len(offspring_fitness)
    assert result.min() >= 0
    assert len(result) == len(parent_fitness)
    np.testing.assert_array_equal(result, expected)

@pytest.mark.parametrize("parent_fitness", [example_fitness])
@pytest.mark.parametrize("offspring_fitness", [
    offspring_fitness_better,
    offspring_fitness_worse,
    offspring_fitness_equal,
    offspring_fitness_mixed,
])
@pytest.mark.parametrize("p", [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
def test_prob_one_to_one(parent_fitness, offspring_fitness, p):
    result = prob_one_to_one(parent_fitness, offspring_fitness, p)
    assert result.max() < len(parent_fitness) + len(offspring_fitness)
    assert result.min() >= 0
    assert len(result) == len(parent_fitness)

@pytest.mark.parametrize("parent_fitness", [example_fitness])
@pytest.mark.parametrize("offspring_fitness", [
    offspring_fitness_better,
    offspring_fitness_worse,
    offspring_fitness_equal,
    offspring_fitness_mixed
])
@pytest.mark.parametrize("amount", [0, 1, 5, 8, 10])
def test_elitism(parent_fitness, offspring_fitness, amount):
    result = elitism(parent_fitness, offspring_fitness, amount)
    assert result.max() < len(parent_fitness) + len(offspring_fitness)
    assert result.min() >= 0
    assert len(result) == len(parent_fitness)
    assert np.all(result[:amount] < len(parent_fitness))
    assert np.all(result[amount:] >= len(parent_fitness))
    np.testing.assert_array_equal(result[:amount], np.argsort(parent_fitness)[::-1][:amount])

# @pytest.mark.parametrize("parent_fitness", [example_fitness])
# @pytest.mark.parametrize("offspring_fitness", [
#     offspring_fitness_better,
#     offspring_fitness_worse,
#     offspring_fitness_equal,
#     offspring_fitness_mixed
# ])
# @pytest.mark.parametrize("amount", [0, 1, 5, 8, 10])
# def test_cond_elitism(parent_fitness, offspring_fitness, amount):
#     result = cond_elitism(parent_fitness, offspring_fitness, amount)
#     assert result.max() < len(parent_fitness) + len(offspring_fitness)
#     assert result.min() >= 0
#     assert len(result) == len(parent_fitness)
#     assert np.all(result[:amount] < len(parent_fitness))
#     assert np.all(result[amount:] >= len(parent_fitness))
#     np.testing.assert_array_equal(result[:amount], np.argsort(parent_fitness)[::-1][:amount])

@pytest.mark.parametrize("parent_fitness", [example_fitness])
@pytest.mark.parametrize("offspring_fitness", [
    offspring_fitness_better,
    offspring_fitness_worse,
    offspring_fitness_equal,
    offspring_fitness_mixed
])
@pytest.mark.parametrize("amount", [0, 1, 5, 8, 10])
def test_lamb_plus_mu(parent_fitness, offspring_fitness, amount):
    result = lamb_plus_mu(parent_fitness, offspring_fitness)
    assert result.max() < len(parent_fitness) + len(offspring_fitness)
    assert result.min() >= 0
    assert len(result) == len(parent_fitness)

@pytest.mark.parametrize("parent_fitness", [example_fitness])
@pytest.mark.parametrize("offspring_fitness", [
    offspring_fitness_better,
    offspring_fitness_worse,
    offspring_fitness_equal,
    offspring_fitness_mixed
])
@pytest.mark.parametrize("amount", [0, 1, 5, 8, 10])
def test_lamb_comma_mu(parent_fitness, offspring_fitness, amount):
    result = lamb_comma_mu(parent_fitness, offspring_fitness)
    assert result.max() < len(parent_fitness) + len(offspring_fitness)
    assert result.min() >= len(parent_fitness)
    assert len(result) == len(parent_fitness)