import pytest
import numpy as np
import metaheuristic_designer as mhd

from metaheuristic_designer import ObjectiveFromLambda, Individual
from metaheuristic_designer.multiObjfunc import CombinedMultiObjectiveFunc
from metaheuristic_designer.selectionMethods.survivor_selection_multi_functions import (
    non_dominated_ranking,
    crowding_distance_selection,
    non_dominated_sorting
)
# from metaheuristic_designer.benchmarks import Sphere
# from metaheuristic_designer

def func1(x):
    return (x**2).sum()

def func2(x):
    return ((x-2)**2).sum()


objfunc1_single = ObjectiveFromLambda(func1, vecsize=1, mode="min")
objfunc2_single = ObjectiveFromLambda(func2, vecsize=1, mode="min")
objfunc = CombinedMultiObjectiveFunc([objfunc1_single, objfunc2_single])

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


def test_crowding_distance():
    pass

def test_nondom_sorting():
    pass