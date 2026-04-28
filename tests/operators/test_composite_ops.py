import pytest

import numpy as np
from metaheuristic_designer import Population
from metaheuristic_designer.operators import VectorOperator, CompositeOperator, BranchOperator, MaskedOperator
from metaheuristic_designer.benchmarks.benchmark_funcs import Sphere
from metaheuristic_designer.initializers import UniformInitializer
import metaheuristic_designer as mhd

mhd.reset_seed(0)

pop_size = 100
example_population1 = Population(Sphere(3), mhd.RAND_GEN.uniform(-100, 100, (pop_size, 3)))
example_population2 = Population(Sphere(20), mhd.RAND_GEN.uniform(-100, 100, (pop_size, 20)))
example_population3 = Population(Sphere(100), mhd.RAND_GEN.uniform(-100, 100, (pop_size, 100)))


@pytest.mark.parametrize("population", [example_population1])
@pytest.mark.parametrize(
    "op_list, args",
    [
        (
            [VectorOperator("dummy", {"F": 1}), VectorOperator("dummy", {"F": 2})],
            {"p": 0.2},
        ),
        (
            [VectorOperator("dummy", {"F": 1}), VectorOperator("dummy", {"F": 2})],
            {"weights": [0.2, 0.8]},
        ),
        (
            [
                VectorOperator("dummy", {"F": 1}),
                VectorOperator("dummy", {"F": 2}),
                VectorOperator("dummy", {"F": 3}),
            ],
            {"weights": [0.2, 0.4, 0.4]},
        ),
    ],
)
def test_branch_op(population, op_list, args):
    pop_init = UniformInitializer(population.vec_size, 0, 1, pop_size)
    operator = BranchOperator(op_list, "random", params=args)

    new_population = operator.evolve(population, pop_init)
    # assert type(new_indiv.genotype) == np.ndarray
    # assert np.all(new_indiv.genotype == new_indiv.genotype[0])
    # assert new_indiv.genotype[0] != 0


@pytest.mark.parametrize("population", [example_population1])
@pytest.mark.parametrize(
    "op_list, expected_val",
    [
        ([VectorOperator("dummy", {"F": 1}), VectorOperator("dummy", {"F": 2})], 2),
        ([VectorOperator("dummy", {"F": 1}), VectorOperator("dummy", {"F": 23})], 23),
        (
            [
                VectorOperator("dummy", {"F": 1}),
                VectorOperator("dummy", {"F": 2}),
                VectorOperator("dummy", {"F": 3}),
            ],
            3,
        ),
    ],
)
def test_sequence_op(population, op_list, expected_val):
    pop_init = UniformInitializer(population.vec_size, 0, 1, pop_size)
    operator = CompositeOperator(op_list)

    new_population = operator.evolve(population, pop_init)
    assert isinstance(new_population.genotype_matrix, np.ndarray)
    assert np.all(new_population.genotype_matrix == expected_val)


@pytest.mark.parametrize("population", [example_population1])
@pytest.mark.parametrize(
    "op_list, values",
    [
        ([VectorOperator("dummy", {"F": 1}), VectorOperator("dummy", {"F": 2})], (1, 2)),
        ([VectorOperator("dummy", {"F": 1}), VectorOperator("dummy", {"F": 23})], (1, 23)),
        (
            [
                VectorOperator("dummy", {"F": 1}),
                VectorOperator("dummy", {"F": 2}),
                VectorOperator("dummy", {"F": 3}),
            ],
            (1, 2, 3),
        ),
    ],
)
def test_pick_op(population, op_list, values):
    pop_init = UniformInitializer(population.vec_size, 0, 1, pop_size)
    operator = BranchOperator(op_list, method="pick", params={})
    for i, _ in enumerate(op_list):
        operator.chosen_idx = i
        new_population = operator.evolve(population, pop_init)
        assert isinstance(new_population.genotype_matrix, np.ndarray)
        assert np.all(new_population.genotype_matrix == values[i])


@pytest.mark.parametrize("population", [example_population1])
@pytest.mark.parametrize(
    "op_list, values",
    [
        ([VectorOperator("dummy", {"F": 1}), VectorOperator("dummy", {"F": 2})], (1, 2)),
        ([VectorOperator("dummy", {"F": 1}), VectorOperator("dummy", {"F": 23})], (1, 23)),
        (
            [
                VectorOperator("dummy", {"F": 1}),
                VectorOperator("dummy", {"F": 2}),
                VectorOperator("dummy", {"F": 3}),
            ],
            (1, 2, 3),
        ),
    ],
)
@pytest.mark.parametrize("mask", [np.zeros(100), np.ones(100), ((np.arange(100) % 2) == 0).astype(int)])
def test_split_op(population, op_list, values, mask):
    pop_init = UniformInitializer(population.vec_size, 0, 1, pop_size)
    mask = mask[:population.vec_size]
    operator = MaskedOperator(op_list, params={"mask": mask})
    new_population = operator.evolve(population, pop_init)

    assert isinstance(new_population.genotype_matrix, np.ndarray)

    for idx, val in enumerate(values):
        if np.any(mask == idx):
            assert np.all(new_population.genotype_matrix[:, mask == idx] == val)
