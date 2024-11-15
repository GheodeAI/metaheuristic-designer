import pytest

import numpy as np
from copy import copy 
from metaheuristic_designer import Individual
from metaheuristic_designer.operators import OperatorVector, vector_ops_map
from metaheuristic_designer.benchmarks.benchmark_funcs import Sphere
from metaheuristic_designer.initializers import UniformVectorInitializer
import metaheuristic_designer as mhd

mhd.reset_seed(0)

vector_ops = [i for i in vector_ops_map.keys() if i not in ["mutate1sigma", "mutatensigmas", "samplesigma"]]

pop_size = 100

example_populaton1 = [Individual(Sphere(3), np.random.uniform(-100, 100, 3)) for i in range(pop_size)]
example_populaton2 = [Individual(Sphere(20), np.random.uniform(-100, 100, 20)) for i in range(pop_size)]
example_populaton3 = [Individual(Sphere(100), np.random.uniform(-100, 100, 100)) for i in range(pop_size)]


def test_errors():
    with pytest.raises(ValueError):
        operator = OperatorVector("not_a_method")


@pytest.mark.parametrize("population", [example_populaton1, example_populaton2, example_populaton3])
@pytest.mark.parametrize("op_method", vector_ops)
def test_basic_working(population, op_method):
    pop_init = UniformVectorInitializer(population[0].genotype.size, 0, 1, pop_size)
    operator = OperatorVector(op_method, "default")

    if op_method in ["xor"]:
        population = copy(population)
        for idx, indiv in enumerate(population):
            population[idx].genotype = indiv.genotype.astype(int)

    indiv = population[0]
    new_population = operator.evolve(population, indiv.objfunc, pop_init, indiv)
    assert type(new_population) == list
    assert type(new_population[0].genotype) == np.ndarray

    # Test when global best is not defined
    # new_population = operator.evolve(population, indiv.objfunc, pop_init)
    # assert type(new_population) == list
    # assert type(new_population[0].genotype) == np.ndarray


# @pytest.mark.parametrize("population", [example_populaton1, example_populaton2, example_populaton3])
# @pytest.mark.parametrize(
#     "distrib, params",
#     [
#         ("uniform", {"max": 0, "max": 1}),
#         ("gauss", {"loc": 2, "scale": 1}),
#         ("cauchy", {"loc": 2, "scale": 1}),
#         ("laplace", {"loc": 2, "scale": 1}),
#         ("gamma", {"a": 0.5}),
#         ("exp", {}),
#         ("levystable", {"a": 1, "b": 1, "loc": 2, "scale": 1}),
#     ],
# )
# @pytest.mark.parametrize("n", [1, 3, 6, 15])
# def test_mutnoise(population, distrib, params, n):
#     pop_init = UniformVectorInitializer(population[0].genotype.size, 0, 1, pop_size)
#     params["distrib"] = distrib
#     params["N"] = n
#     operator = OperatorVector("MutNoise", params)

#     indiv = population[0]
#     new_indiv = operator.evolve(indiv, population, indiv.objfunc, indiv, pop_init)
#     assert type(new_indiv.genotype) == np.ndarray


# @pytest.mark.parametrize("population", [example_populaton2, example_populaton3])
# @pytest.mark.parametrize(
#     "distrib, params",
#     [
#         ("uniform", {"max": 0, "max": 1}),
#         ("gauss", {"loc": 2, "scale": 1}),
#         ("cauchy", {"loc": 2, "scale": 1}),
#         ("laplace", {"loc": 2, "scale": 1}),
#         ("gamma", {"a": 0.5}),
#         ("exp", {}),
#         ("levystable", {"a": 1, "b": 1, "loc": 2, "scale": 1}),
#     ],
# )
# @pytest.mark.parametrize("n", [1, 3, 6, 15])
# def test_mutsample(population, distrib, params, n):
#     pop_init = UniformVectorInitializer(population[0].genotype.size, 0, 1, pop_size)
#     params["distrib"] = distrib
#     params["N"] = n
#     operator = OperatorVector("MutSample", params)

#     indiv = population[0]
#     new_indiv = operator.evolve(indiv, population, indiv.objfunc, indiv, pop_init)
#     assert type(new_indiv.genotype) == np.ndarray


# @pytest.mark.parametrize("population", [example_populaton2, example_populaton3])
# @pytest.mark.parametrize(
#     "distrib, params",
#     [
#         ("uniform", {"max": 0, "max": 1}),
#         ("gauss", {"loc": 2, "scale": 1}),
#         ("cauchy", {"loc": 2, "scale": 1}),
#         ("laplace", {"loc": 2, "scale": 1}),
#         ("gamma", {"a": 0.5}),
#         ("exp", {}),
#         ("levystable", {"a": 1, "b": 1, "loc": 2, "scale": 1}),
#     ],
# )
# def test_randnoise(population, distrib, params):
#     pop_init = UniformVectorInitializer(population[0].genotype.size, 0, 1, pop_size)
#     params["distrib"] = distrib
#     operator = OperatorVector("RandNoise", params)

#     indiv = population[0]
#     new_indiv = operator.evolve(indiv, population, indiv.objfunc, indiv, pop_init)
#     assert type(new_indiv.genotype) == np.ndarray


# @pytest.mark.parametrize("population", [example_populaton1, example_populaton2, example_populaton3])
# @pytest.mark.parametrize(
#     "distrib, params",
#     [
#         ("uniform", {"max": 0, "max": 1}),
#         ("gauss", {"loc": 2, "scale": 1}),
#         ("cauchy", {"loc": 2, "scale": 1}),
#         ("laplace", {"loc": 2, "scale": 1}),
#         ("gamma", {"a": 0.5}),
#         ("exp", {}),
#         ("levystable", {"a": 1, "b": 1, "loc": 2, "scale": 1}),
#     ],
# )
# def test_randsample(population, distrib, params):
#     pop_init = UniformVectorInitializer(population[0].genotype.size, 0, 1, pop_size)
#     params["distrib"] = distrib
#     operator = OperatorVector("RandSample", params)

#     indiv = population[0]
#     new_indiv = operator.evolve(indiv, population, indiv.objfunc, indiv, pop_init)
#     assert type(new_indiv.genotype) == np.ndarray
