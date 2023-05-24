import pytest

import numpy as np
from pyevolcomp import Individual
from pyevolcomp.Operators import OperatorInt, int_ops_map
from pyevolcomp.benchmarks.benchmark_funcs import Sphere
from pyevolcomp.Initializers import UniformVectorInitializer

pop_size = 100

example_populaton1 = [Individual(Sphere(3), np.random.randint(-100, 100, 3)) for i in range(pop_size)]
example_populaton2 = [Individual(Sphere(20), np.random.randint(-100, 100, 20)) for i in range(pop_size)]
example_populaton3 = [Individual(Sphere(100), np.random.randint(-100, 100, 100)) for i in range(pop_size)]

@pytest.mark.parametrize("population", [example_populaton1, example_populaton2, example_populaton3])
@pytest.mark.parametrize("op_method", int_ops_map.keys())
def test_basic_working(population, op_method):
    pop_init = UniformVectorInitializer(population[0].genotype.size, 0, 1, pop_size)
    operator = OperatorInt(op_method)

    indiv = population[0]
    new_indiv = operator.evolve(indiv, population, indiv.objfunc, indiv, pop_init)
    assert type(new_indiv.genotype) == np.ndarray