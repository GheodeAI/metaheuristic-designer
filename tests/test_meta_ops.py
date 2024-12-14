import pytest

import numpy as np
from metaheuristic_designer import Population
from metaheuristic_designer.operators import OperatorMeta, meta_ops_map, OperatorVector
from metaheuristic_designer.benchmarks.benchmark_funcs import Sphere
from metaheuristic_designer.initializers import UniformVectorInitializer
import metaheuristic_designer as mhd

mhd.reset_seed(0)

pop_size = 100
example_population1 = Population(Sphere(3), mhd.RAND_GEN.uniform(-100, 100, (pop_size, 3)))
example_population2 = Population(Sphere(20), mhd.RAND_GEN.uniform(-100, 100, (pop_size, 20)))
example_population3 = Population(Sphere(100), mhd.RAND_GEN.uniform(-100, 100, (pop_size, 100)))

def test_errors():
    with pytest.raises(ValueError):
        operator = OperatorMeta("not_a_method", [])


@pytest.mark.parametrize("population", [example_population1])
@pytest.mark.parametrize(
    "op_list, args",
    [
        (
            [OperatorVector("dummy", {"F": 1}), OperatorVector("dummy", {"F": 2})],
            {"p": 0.2},
        ),
        (
            [OperatorVector("dummy", {"F": 1}), OperatorVector("dummy", {"F": 2})],
            {"weights": [0.2, 0.8]},
        ),
        (
            [
                OperatorVector("dummy", {"F": 1}),
                OperatorVector("dummy", {"F": 2}),
                OperatorVector("dummy", {"F": 3}),
            ],
            {"weights": [0.2, 0.4, 0.4]},
        ),
    ],
)
def test_branch_op(population, op_list, args):
    pop_init = UniformVectorInitializer(population.vec_size, 0, 1, pop_size)
    operator = OperatorMeta("branch", op_list, args)

    new_population = operator.evolve(population, pop_init)
    # assert type(new_indiv.genotype) == np.ndarray
    # assert np.all(new_indiv.genotype == new_indiv.genotype[0])
    # assert new_indiv.genotype[0] != 0


@pytest.mark.parametrize("population", [example_population1])
@pytest.mark.parametrize(
    "op_list, expected_val",
    [
        ([OperatorVector("dummy", {"F": 1}), OperatorVector("dummy", {"F": 2})], 2),
        ([OperatorVector("dummy", {"F": 1}), OperatorVector("dummy", {"F": 23})], 23),
        (
            [
                OperatorVector("dummy", {"F": 1}),
                OperatorVector("dummy", {"F": 2}),
                OperatorVector("dummy", {"F": 3}),
            ],
            3,
        ),
    ],
)
def test_sequence_op(population, op_list, expected_val):
    pop_init = UniformVectorInitializer(population.vec_size, 0, 1, pop_size)
    operator = OperatorMeta("sequence", op_list)

    new_population = operator.evolve(population, pop_init)
    # assert type(new_indiv.genotype) == np.ndarray
    # assert np.all(new_indiv.genotype == new_indiv.genotype[0])
    # assert new_indiv.genotype[0] == expected_val


@pytest.mark.parametrize("population", [example_population1])
@pytest.mark.parametrize(
    "op_list, values",
    [
        ([OperatorVector("dummy", {"F": 1}), OperatorVector("dummy", {"F": 2})], (0, 1)),
        ([OperatorVector("dummy", {"F": 1}), OperatorVector("dummy", {"F": 23})], (1, 23)),
        (
            [
                OperatorVector("dummy", {"F": 1}),
                OperatorVector("dummy", {"F": 2}),
                OperatorVector("dummy", {"F": 3}),
            ],
            (1, 2, 3),
        ),
    ],
)
def test_pick_op(population, op_list, values):
    pop_init = UniformVectorInitializer(population.vec_size, 0, 1, pop_size)
    operator = OperatorMeta("pick", op_list, {})
    for i, _ in enumerate(op_list):
        operator.chosen_idx = i
        new_population = operator.evolve(population, pop_init)
        # new_indiv = operator.evolve(example_individual, [example_individual], None, example_individual, pop_init)
        # assert type(new_indiv.genotype) == np.ndarray
        # assert np.all(new_indiv.genotype == values[i])


@pytest.mark.parametrize("population", [example_population1])
@pytest.mark.parametrize(
    "op_list, values",
    [
        ([OperatorVector("dummy", {"F": 1}), OperatorVector("dummy", {"F": 2})], (1, 2)),
        ([OperatorVector("dummy", {"F": 1}), OperatorVector("dummy", {"F": 23})], (1, 23)),
        (
            [
                OperatorVector("dummy", {"F": 1}),
                OperatorVector("dummy", {"F": 2}),
                OperatorVector("dummy", {"F": 3}),
            ],
            (1, 2, 3),
        ),
    ],
)
@pytest.mark.parametrize("mask", [np.zeros(100), np.ones(100), ((np.arange(100) % 2) == 0).astype(int)])
def test_split_op(population, op_list, values, mask):
    pop_init = UniformVectorInitializer(population.vec_size, 0, 1, pop_size)
    mask = mask[:population.vec_size]
    operator = OperatorMeta("split", op_list, {"mask": mask})
    new_population = operator.evolve(population, pop_init)

    # new_indiv = operator.evolve(example_individual, [example_individual], None, example_individual, pop_init)
    # assert type(new_indiv.genotype) == np.ndarray

    # for idx, val in enumerate(values):
    #     if np.any(mask == idx):
    #         assert np.all(new_indiv.genotype[mask == idx] == val)
