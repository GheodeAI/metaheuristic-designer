import pytest

import numpy as np
from metaheuristic_designer import Individual
from metaheuristic_designer.Operators import OperatorPerm, perm_ops_map
from metaheuristic_designer.benchmarks.benchmark_funcs import Sphere
from metaheuristic_designer.Initializers import PermInitializer

perm_ops = [i for i in perm_ops_map.keys()]

pop_size = 100

example_populaton1 = [Individual(Sphere(3), np.arange(3)) for i in range(pop_size)]
example_populaton2 = [Individual(Sphere(20), np.arange(20)) for i in range(pop_size)]
example_populaton3 = [Individual(Sphere(100), np.arange(100)) for i in range(pop_size)]

@pytest.mark.parametrize("population", [example_populaton1, example_populaton2, example_populaton3])
@pytest.mark.parametrize("op_method", perm_ops)
def test_basic_working(population, op_method):
    pop_init = PermInitializer(population[0].genotype.size, pop_size)
    operator = OperatorPerm(op_method, "default")

    indiv = population[0]
    new_indiv = operator.evolve(indiv, population, indiv.objfunc, indiv, pop_init)
    assert type(new_indiv.genotype) == np.ndarray