import pytest

import numpy as np
from metaheuristic_designer import Individual
from metaheuristic_designer.operators import OperatorMeta, meta_ops_map, OperatorReal
from metaheuristic_designer.benchmarks.benchmark_funcs import Sphere
from metaheuristic_designer.initializers import UniformVectorInitializer
import metaheuristic_designer as mhd

mhd.reset_seed(0)

pop_size = 100

example_individual = Individual(None, np.zeros(100))
pop_init = UniformVectorInitializer(100, 0, 1, pop_size)


def test_errors():
    with pytest.raises(ValueError):
        operator = OperatorMeta("not_a_method", [])


@pytest.mark.parametrize("indiv", [example_individual])
@pytest.mark.parametrize(
    "op_list, args",
    [
        (
            [OperatorReal("dummy", {"F": 1}), OperatorReal("dummy", {"F": 2})],
            {"p": 0.2},
        ),
        (
            [OperatorReal("dummy", {"F": 1}), OperatorReal("dummy", {"F": 2})],
            {"weights": [0.2, 0.8]},
        ),
        (
            [
                OperatorReal("dummy", {"F": 1}),
                OperatorReal("dummy", {"F": 2}),
                OperatorReal("dummy", {"F": 3}),
            ],
            {"weights": [0.2, 0.4, 0.4]},
        ),
    ],
)
def test_branch_op(indiv, op_list, args):
    operator = OperatorMeta("branch", op_list, args)

    new_indiv = operator.evolve(
        example_individual, [example_individual], None, example_individual, pop_init
    )
    assert type(new_indiv.genotype) == np.ndarray
    assert np.all(new_indiv.genotype == new_indiv.genotype[0])
    assert new_indiv.genotype[0] != 0


@pytest.mark.parametrize("indiv", [example_individual])
@pytest.mark.parametrize(
    "op_list, expected_val",
    [
        ([OperatorReal("dummy", {"F": 1}), OperatorReal("dummy", {"F": 2})], 2),
        ([OperatorReal("dummy", {"F": 1}), OperatorReal("dummy", {"F": 23})], 23),
        (
            [
                OperatorReal("dummy", {"F": 1}),
                OperatorReal("dummy", {"F": 2}),
                OperatorReal("dummy", {"F": 3}),
            ],
            3,
        ),
    ],
)
def test_sequence_op(indiv, op_list, expected_val):
    operator = OperatorMeta("sequence", op_list)

    new_indiv = operator.evolve(
        example_individual, [example_individual], None, example_individual, pop_init
    )
    assert type(new_indiv.genotype) == np.ndarray
    assert np.all(new_indiv.genotype == new_indiv.genotype[0])
    assert new_indiv.genotype[0] == expected_val


@pytest.mark.parametrize("indiv", [example_individual])
@pytest.mark.parametrize(
    "op_list, values",
    [
        ([OperatorReal("dummy", {"F": 1}), OperatorReal("dummy", {"F": 2})], (1, 2)),
        ([OperatorReal("dummy", {"F": 1}), OperatorReal("dummy", {"F": 23})], (1, 23)),
        (
            [
                OperatorReal("dummy", {"F": 1}),
                OperatorReal("dummy", {"F": 2}),
                OperatorReal("dummy", {"F": 3}),
            ],
            (1, 2, 3),
        ),
    ],
)
def test_pick_op(indiv, op_list, values):
    operator = OperatorMeta("pick", op_list, {})
    for i, _ in enumerate(op_list):
        operator.chosen_idx = i
        new_indiv = operator.evolve(
            example_individual, [example_individual], None, example_individual, pop_init
        )
        assert type(new_indiv.genotype) == np.ndarray
        assert np.all(new_indiv.genotype == values[i])


@pytest.mark.parametrize("indiv", [example_individual])
@pytest.mark.parametrize(
    "op_list, values",
    [
        ([OperatorReal("dummy", {"F": 1}), OperatorReal("dummy", {"F": 2})], (1, 2)),
        ([OperatorReal("dummy", {"F": 1}), OperatorReal("dummy", {"F": 23})], (1, 23)),
        (
            [
                OperatorReal("dummy", {"F": 1}),
                OperatorReal("dummy", {"F": 2}),
                OperatorReal("dummy", {"F": 3}),
            ],
            (1, 2, 3),
        ),
    ],
)
@pytest.mark.parametrize(
    "mask", [np.zeros(100), np.ones(100), (np.arange(100) > 50).astype(int)]
)
def test_split_op(indiv, op_list, values, mask):
    operator = OperatorMeta("split", op_list, {"mask": mask})

    new_indiv = operator.evolve(
        example_individual, [example_individual], None, example_individual, pop_init
    )
    assert type(new_indiv.genotype) == np.ndarray

    for idx, val in enumerate(values):
        if np.any(mask == idx):
            assert np.all(new_indiv.genotype[mask == idx] == val)
