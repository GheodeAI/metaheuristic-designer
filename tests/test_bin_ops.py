import pytest

import numpy as np
from pyevolcomp import Individual
from pyevolcomp.Operators import OperatorBinary, _bin_ops
from pyevolcomp.benchmarks.benchmark_funcs import MaxOnes

pop_size = 100

example_populaton1 = [Individual(MaxOnes(3), np.random.random(3) > 0.5) for i in range(pop_size)]
example_populaton2 = [Individual(MaxOnes(20), np.random.random(20) > 0.5) for i in range(pop_size)]
example_populaton3 = [Individual(MaxOnes(100), np.random.random(100) > 0.5) for i in range(pop_size)]

@pytest.mark.parametrize("population", [example_populaton1, example_populaton2, example_populaton3])
@pytest.mark.parametrize("op_method", _bin_ops)
def test_basic_working(population, op_method):
    operator = OperatorBinary(op_method)

    indiv = population[0]
    new_indiv = operator.evolve(indiv, population, indiv.objfunc, indiv)
    assert type(new_indiv.genotype) == np.ndarray