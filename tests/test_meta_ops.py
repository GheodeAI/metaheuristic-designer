import pytest

import numpy as np
from pyevolcomp import Individual
from pyevolcomp.Operators import OperatorMeta, meta_ops_map, OperatorReal
from pyevolcomp.benchmarks.benchmark_funcs import Sphere

pop_size = 100

example_individual = Individual(None, np.zeros(100))


@pytest.mark.parametrize("indiv", [example_individual])
@pytest.mark.parametrize("op_list, args", [
    ([OperatorReal("dummy", {"F": 1}), OperatorReal("dummy", {"F": 2})], {"p": 0.2}),
    ([OperatorReal("dummy", {"F": 1}), OperatorReal("dummy", {"F": 2})], {"weights": [0.2, 0.8]}),
    ([OperatorReal("dummy", {"F": 1}), OperatorReal("dummy", {"F": 2}), OperatorReal("dummy", {"F": 3})], {"weights": [0.2, 0.4, 0.4]})
])
def test_branch_op(indiv, op_list, args):
    print(op_list)
    print(args)
    operator = OperatorMeta("branch", op_list, args)

    new_indiv = operator.evolve(example_individual, [example_individual], None, example_individual)
    assert type(new_indiv.genotype) == np.ndarray
    assert np.all(new_indiv.genotype == new_indiv.genotype[0])
    assert new_indiv.genotype[0] != 0


@pytest.mark.parametrize("indiv", [example_individual])
@pytest.mark.parametrize("op_list, expected_val", [
    ([OperatorReal("dummy", {"F": 1}), OperatorReal("dummy", {"F": 2})], 2),
    ([OperatorReal("dummy", {"F": 1}), OperatorReal("dummy", {"F": 23})], 23),
    ([OperatorReal("dummy", {"F": 1}), OperatorReal("dummy", {"F": 2}), OperatorReal("dummy", {"F": 3})], 3)
])
def test_sequence_op(indiv, op_list, expected_val):
    operator = OperatorMeta("sequence", op_list)

    new_indiv = operator.evolve(example_individual, [example_individual], None, example_individual)
    assert type(new_indiv.genotype) == np.ndarray
    assert np.all(new_indiv.genotype == new_indiv.genotype[0])
    assert new_indiv.genotype[0] == expected_val