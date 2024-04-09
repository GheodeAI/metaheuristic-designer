import pytest
import numpy as np
import metaheuristic_designer as mhd

from metaheuristic_designer import ObjectiveFromLambda, Individual
from metaheuristic_designer.benchmarks import *
from metaheuristic_designer.multiObjfunc import CombinedMultiObjectiveFunc
from metaheuristic_designer.selectionMethods.survivor_selection_multi_functions import (
    non_dominated_ranking,
    crowding_distance_selection,
    non_dominated_sorting
)

objfunc = Shaffer1()

test_population_arr1 = np.array([
    [-1], # id=0
    [-2], # id=1
    [0],  # id=2
    [1],  # id=3
    [2],  # id=4
    [4],  # id=5
    [10], # id=6
])

test_population1 = [
    Individual(objfunc, vec, individual_id=idx)
    for idx, vec in enumerate(test_population_arr1)
]

test_population_arr2 = np.array([
    [0],    # id=0
    [1],    # id=1
    [2],    # id=2
    [0.75], # id=3
])

test_population2 = [
    Individual(objfunc, vec, individual_id=idx)
    for idx, vec in enumerate(test_population_arr2)
]

test_population_arr3 = np.array([
    [0],   # id=0
    [4],   # id=1
    [10],  # id=2
    [100], # id=3
])

test_population3 = [
    Individual(objfunc, vec, individual_id=idx)
    for idx, vec in enumerate(test_population_arr3)
]

test_ranks_arr1 = [
    [
        [0],     # id=0
        [0.75],  # id=1
        [1],     # id=2
        [1.25],  # id=3
        [2],     # id=4
    ],
    [
        [-0.25], # id=5
        [-0.1],  # id=6
        [-0.01], # id=7
        [3],     # id=8
        [3.2],   # id=9
    ],
    [
        [9.5],   # id=10
        [9.75],  # id=11
        [10],    # id=12
        [11],    # id=13
        [12],    # id=14
    ]
]

test_ranks1 = []
i = 0
for idx_y, rank in enumerate(test_ranks_arr1):
    rank_list = []
    for idx_x, vec in enumerate(rank):
        rank_list.append(Individual(objfunc, np.array(vec), individual_id=i))
        i += 1
    test_ranks1.append(rank_list)


@pytest.mark.parametrize("population, expected_rank_id", [
    (test_population1, [[2, 4, 3], [0, 5], [1], [6]]),
    (test_population2, [[0, 1, 2, 3]]),
    (test_population3, [[0], [1], [2], [3]])
])
def test_nondom_ranking(population, expected_rank_id):
    ranked_pop = non_dominated_ranking(population)
    result_id_list = [[i.id for i in rank] for rank in ranked_pop]
    for exp, res in zip(expected_rank_id, result_id_list):
        assert set(exp) == set(res)

@pytest.mark.parametrize("amount, id_list", [
    (1, [2]),
    (5, [2, 1, 3, 0, 4]),
    (7, [2, 1, 3, 0, 4, 6, 5]),
    (12, [2, 1, 3, 0, 4, 6, 5, 7, 8, 9, 11, 12]),
])
def test_crowding_distance(amount, id_list):
    sorted_individuals = crowding_distance_selection(test_ranks1, amount)
    # print([i.id for i in sorted_individuals])
    # print([i.genotype for i in sorted_individuals])
    sorted_id = [i.id for i in sorted_individuals]
    assert sorted_id == id_list

@pytest.mark.parametrize("population, amount, id_list", [
    # (test_population1, 1, [[2, 4, 3], [0, 5], [1], [6]]),
    # (test_population1, 5, [[2, 4, 3], [0, 5], [1], [6]]),
    (test_population2, 1, [0]),
    (test_population2, 3, [0, 1, 2, 3]),
    (test_population3, 1, [0]),
    (test_population3, 3, [0, 1, 2]),
])
def test_nondom_sorting(population, amount, id_list):
    selected = non_dominated_sorting(population, amount)
    sorted_id = [i.id for i in selected]
    assert sorted_id == id_list
    