import pytest

import numpy as np
from metaheuristic_designer import Individual
from metaheuristic_designer.operators import OperatorReal, real_ops_map
from metaheuristic_designer.benchmarks.benchmark_funcs import Sphere
from metaheuristic_designer.initializers import UniformVectorInitializer
import metaheuristic_designer as mhd

mhd.reset_seed(0)

real_ops = [
    i for i in real_ops_map.keys() if i not in ["mutate1sigma", "mutatensigmas"]
]

pop_size = 100

example_populaton1 = [
    Individual(Sphere(3), np.random.uniform(-100, 100, 3)) for i in range(pop_size)
]
example_populaton2 = [
    Individual(Sphere(20), np.random.uniform(-100, 100, 20)) for i in range(pop_size)
]
example_populaton3 = [
    Individual(Sphere(100), np.random.uniform(-100, 100, 100)) for i in range(pop_size)
]


def test_errors():
    with pytest.raises(ValueError):
        operator = OperatorReal("not_a_method")


@pytest.mark.parametrize(
    "population", [example_populaton1, example_populaton2, example_populaton3]
)
@pytest.mark.parametrize("op_method", real_ops)
def test_basic_working(population, op_method):
    pop_init = UniformVectorInitializer(population[0].genotype.size, 0, 1, pop_size)
    operator = OperatorReal(op_method, "default")

    indiv = population[0]
    new_indiv = operator.evolve(indiv, population, indiv.objfunc, indiv, pop_init)
    assert type(new_indiv.genotype) == np.ndarray

    # Test when global best is not defined
    new_indiv = operator.evolve(indiv, population, indiv.objfunc, None, pop_init)
    assert type(new_indiv.genotype) == np.ndarray
