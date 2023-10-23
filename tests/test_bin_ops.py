import pytest

import numpy as np
from metaheuristic_designer import Individual
from metaheuristic_designer.operators import OperatorBinary, bin_ops_map
from metaheuristic_designer.benchmarks.benchmark_funcs import MaxOnes
from metaheuristic_designer.initializers import UniformVectorInitializer
import metaheuristic_designer as mhd

mhd.reset_seed(0)

pop_size = 100

example_populaton1 = [
    Individual(MaxOnes(3), np.random.random(3) > 0.5) for i in range(pop_size)
]
example_populaton2 = [
    Individual(MaxOnes(20), np.random.random(20) > 0.5) for i in range(pop_size)
]
example_populaton3 = [
    Individual(MaxOnes(100), np.random.random(100) > 0.5) for i in range(pop_size)
]


def test_errors():
    with pytest.raises(ValueError):
        operator = OperatorBinary("not_a_method")


@pytest.mark.parametrize(
    "population", [example_populaton1, example_populaton2, example_populaton3]
)
@pytest.mark.parametrize("op_method", bin_ops_map.keys())
def test_basic_working(population, op_method):
    pop_init = UniformVectorInitializer(population[0].genotype.size, 0, 1, pop_size)
    operator = OperatorBinary(op_method, "default")

    indiv = population[0]
    new_indiv = operator.evolve(indiv, population, indiv.objfunc, indiv, pop_init)
    assert type(new_indiv.genotype) == np.ndarray
