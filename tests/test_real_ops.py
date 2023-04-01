import pytest

import numpy as np
from pyevolcomp import Individual
from pyevolcomp.Operators import OperatorReal, _real_ops
from pyevolcomp.benchmarks.benchmark_funcs import Sphere

_real_ops = [i for i in _real_ops if i not in ["mutate1sigma", "mutatensigmas"]]

pop_size = 100

example_populaton1 = [Individual(Sphere(3), np.random.uniform(-100, 100, 3)) for i in range(pop_size)]
example_populaton2 = [Individual(Sphere(20), np.random.uniform(-100, 100, 20)) for i in range(pop_size)]
example_populaton3 = [Individual(Sphere(100), np.random.uniform(-100, 100, 100)) for i in range(pop_size)]

@pytest.mark.parametrize("population", [example_populaton1, example_populaton2, example_populaton3])
@pytest.mark.parametrize("op_method", _real_ops)
def test_basic_working(population, op_method):
    operator = OperatorReal(op_method)

    indiv = population[0]
    new_indiv = operator.evolve(indiv, population, indiv.objfunc, indiv)
    assert type(new_indiv.genotype) == np.ndarray