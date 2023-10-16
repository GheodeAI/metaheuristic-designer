import pytest

import numpy as np
from metaheuristic_designer import Individual
from metaheuristic_designer.initializers import *
import metaheuristic_designer as mhd

mhd.reset_seed(0)

pop_size = 100

example_populaton1 = [
    Individual(None, np.random.uniform(-100, 100, 3)) for i in range(100)
]
example_populaton2 = [
    Individual(None, np.random.uniform(-100, 100, 20)) for i in range(100)
]
example_populaton3 = [
    Individual(None, np.random.uniform(-100, 100, 100)) for i in range(100)
]


@pytest.mark.parametrize("vec_size", [1, 2, 10, 100])
@pytest.mark.parametrize("min_val, max_val", [(0, 1), (-1, 1), (-100, 2), (2, 24)])
@pytest.mark.parametrize("pop_size", [1, 10, 100])
def test_uniform_vec_init(vec_size, min_val, max_val, pop_size):
    pop_init = UniformVectorInitializer(vec_size, min_val, max_val, pop_size)

    for _ in range(30):
        rand_inidv = pop_init.generate_random(None)
        assert rand_inidv.genotype.max() <= max_val
        assert rand_inidv.genotype.min() >= min_val
        assert rand_inidv.genotype.size == vec_size

        rand_inidv = pop_init.generate_individual(None)
        assert rand_inidv.genotype.max() <= max_val
        assert rand_inidv.genotype.min() >= min_val
        assert rand_inidv.genotype.size == vec_size

    rand_pop = pop_init.generate_population(None)
    assert len(rand_pop) == pop_size

    for indiv in rand_pop:
        assert indiv.genotype.max() <= max_val
        assert indiv.genotype.min() >= min_val
        assert indiv.genotype.size == vec_size


@pytest.mark.parametrize(
    "min_val, max_val",
    [
        (np.zeros(10), np.ones(10)),
        (np.full(10, -1), np.ones(10)),
        (np.full(10, -100), np.full(10, 2)),
        (np.full(10, 2), np.full(10, 24)),
    ],
)
@pytest.mark.parametrize("pop_size", [1, 10, 100])
def test_uniform_arr_param_vec_init(min_val, max_val, pop_size):
    pop_init = UniformVectorInitializer(10, min_val, max_val, pop_size)

    for _ in range(30):
        rand_inidv = pop_init.generate_random(None)
        assert np.all(rand_inidv.genotype <= max_val)
        assert np.all(rand_inidv.genotype >= min_val)
        assert rand_inidv.genotype.size == 10

        rand_inidv = pop_init.generate_individual(None)
        assert np.all(rand_inidv.genotype <= max_val)
        assert np.all(rand_inidv.genotype >= min_val)
        assert rand_inidv.genotype.size == 10

    rand_pop = pop_init.generate_population(None)
    assert len(rand_pop) == pop_size

    for indiv in rand_pop:
        assert np.all(indiv.genotype <= max_val)
        assert np.all(indiv.genotype >= min_val)
        assert indiv.genotype.size == 10


def test_uniform_int_vec_init():
    pop_init = UniformVectorInitializer(200, 0, 1, 100, dtype=int)
    rand_inidv = pop_init.generate_random(None)
    assert 0 in rand_inidv.genotype
    assert 1 in rand_inidv.genotype


def test_uniform_err_vec_init():
    with pytest.raises(ValueError):
        UniformVectorInitializer(10, np.zeros(10), np.ones(9))

    with pytest.raises(ValueError):
        UniformVectorInitializer(10, np.zeros(9), np.ones(10))


@pytest.mark.parametrize("list_size", [1, 2, 10, 100])
@pytest.mark.parametrize("min_val, max_val", [(0, 1), (-1, 1), (-100, 2), (2, 24)])
@pytest.mark.parametrize("pop_size", [1, 10, 100])
def test_uniform_list_init(list_size, min_val, max_val, pop_size):
    pop_init = UniformListInitializer(list_size, min_val, max_val, pop_size)

    for _ in range(30):
        rand_inidv = pop_init.generate_random(None)
        assert max(rand_inidv.genotype) <= max_val
        assert min(rand_inidv.genotype) >= min_val
        assert len(rand_inidv.genotype) == list_size

        rand_inidv = pop_init.generate_individual(None)
        assert max(rand_inidv.genotype) <= max_val
        assert min(rand_inidv.genotype) >= min_val
        assert len(rand_inidv.genotype) == list_size

    rand_pop = pop_init.generate_population(None)
    assert len(rand_pop) == pop_size

    for indiv in rand_pop:
        assert max(indiv.genotype) <= max_val
        assert min(indiv.genotype) >= min_val
        assert len(indiv.genotype) == list_size


@pytest.mark.parametrize("vec_size", [1, 2, 10, 100])
@pytest.mark.parametrize(
    "mean_val, std_val",
    [
        (0, 1),
        (-1, 1),
        (-100, 2),
        (2, 24),
    ],
)
@pytest.mark.parametrize("pop_size", [1, 10, 100])
def test_gaussian_vec_init(vec_size, mean_val, std_val, pop_size):
    pop_init = GaussianVectorInitializer(vec_size, mean_val, std_val, pop_size)

    for _ in range(30):
        rand_inidv = pop_init.generate_random(None)
        assert rand_inidv.genotype.size == vec_size

        rand_inidv = pop_init.generate_individual(None)
        assert rand_inidv.genotype.size == vec_size

    rand_pop = pop_init.generate_population(None)
    assert len(rand_pop) == pop_size

    for indiv in rand_pop:
        assert indiv.genotype.size == vec_size


@pytest.mark.parametrize("list_size", [1, 2, 10, 100])
@pytest.mark.parametrize("mean_val, std_val", [(0, 1), (-1, 1), (-100, 2), (2, 24)])
@pytest.mark.parametrize("pop_size", [1, 10, 100])
def test_gaussian_list_init(list_size, mean_val, std_val, pop_size):
    pop_init = GaussianListInitializer(list_size, mean_val, std_val, pop_size)

    for _ in range(30):
        rand_inidv = pop_init.generate_random(None)
        assert len(rand_inidv.genotype) == list_size

        rand_inidv = pop_init.generate_individual(None)
        assert len(rand_inidv.genotype) == list_size

    rand_pop = pop_init.generate_population(None)
    assert len(rand_pop) == pop_size

    for indiv in rand_pop:
        assert len(indiv.genotype) == list_size


@pytest.mark.parametrize(
    "min_val, max_val",
    [
        (np.zeros(10), np.ones(10)),
        (np.full(10, -1), np.ones(10)),
        (np.full(10, -100), np.full(10, 2)),
        (np.full(10, 2), np.full(10, 24)),
    ],
)
@pytest.mark.parametrize("pop_size", [1, 10, 100])
def test_gaussian_arr_param_vec_init(min_val, max_val, pop_size):
    pop_init = GaussianVectorInitializer(10, min_val, max_val, pop_size)

    for _ in range(30):
        rand_inidv = pop_init.generate_random(None)
        assert rand_inidv.genotype.size == 10

        rand_inidv = pop_init.generate_individual(None)
        assert rand_inidv.genotype.size == 10

    rand_pop = pop_init.generate_population(None)
    assert len(rand_pop) == pop_size

    for indiv in rand_pop:
        assert indiv.genotype.size == 10


def test_gaussian_int_vec_init():
    pop_init = GaussianVectorInitializer(200, 0, 1, 100, dtype=int)
    rand_inidv = pop_init.generate_random(None)


def test_gaussian_err_vec_init():
    with pytest.raises(ValueError):
        GaussianVectorInitializer(10, np.zeros(10), np.ones(9))

    with pytest.raises(ValueError):
        GaussianVectorInitializer(10, np.zeros(9), np.ones(10))


@pytest.mark.parametrize(
    "population", [example_populaton1, example_populaton2, example_populaton3]
)
def test_direct_initializer(population):
    default_pop_init = GaussianVectorInitializer(population[0].genotype.size, -100, 100)
    pop_init = DirectInitializer(default_pop_init, population)

    ids_in_pop = [i.id for i in population]

    for _ in range(30):
        rand_inidv = pop_init.generate_random(None)
        assert rand_inidv.id not in ids_in_pop

        rand_inidv = pop_init.generate_individual(None)
        assert rand_inidv.id in ids_in_pop

    rand_pop = pop_init.generate_population(None)
    assert len(rand_pop) == pop_size

    for indiv in rand_pop:
        assert indiv.id in ids_in_pop


@pytest.mark.parametrize(
    "population", [example_populaton1, example_populaton2, example_populaton3]
)
def test_seed_prob_initializer(population):
    default_pop_init = GaussianVectorInitializer(population[0].genotype.size, -100, 100)
    pop_init = SeedProbInitializer(default_pop_init, population)

    ids_in_pop = [i.id for i in population]

    for _ in range(30):
        rand_inidv = pop_init.generate_random(None)
        assert rand_inidv.id not in ids_in_pop

        rand_inidv = pop_init.generate_individual(None)

    rand_pop = pop_init.generate_population(None)


@pytest.mark.parametrize(
    "population", [example_populaton1, example_populaton2, example_populaton3]
)
def test_seed_determ_initializer(population):
    default_pop_init = GaussianVectorInitializer(population[0].genotype.size, -100, 100)
    pop_init = SeedDetermInitializer(default_pop_init, population, 4)

    ids_in_pop = [i.id for i in population]

    for _ in range(30):
        rand_inidv = pop_init.generate_random(None)
        assert rand_inidv.id not in ids_in_pop

        rand_inidv = pop_init.generate_individual(None)

    rand_pop = pop_init.generate_population(None)


@pytest.mark.parametrize(
    "population", [example_populaton1, example_populaton2, example_populaton3]
)
def test_seed_determ_null_param_initializer(population):
    default_pop_init = GaussianVectorInitializer(population[0].genotype.size, -100, 100)
    pop_init = SeedDetermInitializer(default_pop_init, population)


@pytest.mark.parametrize("vec_size", [1, 2, 10, 100])
@pytest.mark.parametrize("pop_size", [1, 10, 100])
def test_lambda_init(vec_size, pop_size):
    generator = lambda: np.zeros(vec_size)
    pop_init = LambdaInitializer(generator, pop_size)

    for _ in range(30):
        rand_inidv = pop_init.generate_random(None)
        assert rand_inidv.genotype.size == vec_size

        rand_inidv = pop_init.generate_individual(None)
        assert rand_inidv.genotype.size == vec_size

    rand_pop = pop_init.generate_population(None)
    assert len(rand_pop) == pop_size

    for indiv in rand_pop:
        assert indiv.genotype.size == vec_size


@pytest.mark.parametrize("vec_size", [2, 10, 100])
@pytest.mark.parametrize("pop_size", [1, 10, 100])
def test_perm_init(vec_size, pop_size):
    pop_init = PermInitializer(vec_size, pop_size)

    for _ in range(30):
        rand_inidv = pop_init.generate_random(None)
        assert np.all(np.isin(np.arange(vec_size), rand_inidv.genotype))

        rand_inidv = pop_init.generate_individual(None)
        assert np.all(np.isin(np.arange(vec_size), rand_inidv.genotype))

    rand_pop = pop_init.generate_population(None)
    assert len(rand_pop) == pop_size

    for indiv in rand_pop:
        assert np.all(np.isin(np.arange(vec_size), rand_inidv.genotype))
