import pytest

import numpy as np
from metaheuristic_designer import Population
from metaheuristic_designer.operators import OperatorPerm, perm_ops_map
from metaheuristic_designer.benchmarks.benchmark_funcs import Sphere
from metaheuristic_designer.initializers import PermInitializer
import metaheuristic_designer as mhd

mhd.reset_seed(0)

perm_ops = [i for i in perm_ops_map.keys()]

pop_size = 100
example_population1 = Population(Sphere(3), np.tile(np.arange(3), (pop_size, 1)))
example_population2 = Population(Sphere(20), np.tile(np.arange(20), (pop_size, 1)))
example_population3 = Population(Sphere(100), np.tile(np.arange(100), (pop_size, 1)))


@pytest.mark.parametrize("population", [example_population1, example_population2, example_population3])
@pytest.mark.parametrize("op_method", perm_ops)
def test_basic_working(population, op_method):
    pop_init = PermInitializer(population.vec_size, pop_size)
    operator = OperatorPerm(op_method, "default")

    new_population = operator.evolve(population, pop_init)
    assert isinstance(new_population, Population)
