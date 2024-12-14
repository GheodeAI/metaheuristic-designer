import pytest

import numpy as np
from metaheuristic_designer import Population 
from metaheuristic_designer.initializers import *
import metaheuristic_designer as mhd

mhd.reset_seed(0)

pop_size = 100

n_indiv = 10
n_components = 5
sample_pop1 = np.tile(np.arange(n_components), n_indiv).reshape((n_indiv, n_components)) + 10 * np.arange(n_indiv).reshape((n_indiv, 1))

@pytest.mark.parametrize("vec_size", [1, 2, 10, 100])
@pytest.mark.parametrize("min_val, max_val", [(0, 1), (-1, 1), (-100, 2), (2, 24)])
@pytest.mark.parametrize("pop_size", [1, 10, 100])
def test_uniform_vec_init(vec_size, min_val, max_val, pop_size):
    pop_init = UniformVectorInitializer(vec_size, min_val, max_val, pop_size)

    for _ in range(30):
        rand_inidv = pop_init.generate_random()
        assert rand_inidv.max() <= max_val
        assert rand_inidv.min() >= min_val
        assert rand_inidv.size == vec_size

        rand_inidv = pop_init.generate_individual()
        assert rand_inidv.max() <= max_val
        assert rand_inidv.min() >= min_val
        assert rand_inidv.size == vec_size

    rand_pop = pop_init.generate_population(None)
    assert len(rand_pop) == pop_size

    for indiv in rand_pop:
        assert indiv.max() <= max_val
        assert indiv.min() >= min_val
        assert indiv.size == vec_size


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
        rand_inidv = pop_init.generate_random()
        assert np.all(rand_inidv <= max_val)
        assert np.all(rand_inidv >= min_val)
        assert rand_inidv.size == 10

        rand_inidv = pop_init.generate_individual()
        assert np.all(rand_inidv <= max_val)
        assert np.all(rand_inidv >= min_val)
        assert rand_inidv.size == 10

    rand_pop = pop_init.generate_population(None)
    assert len(rand_pop) == pop_size

    for indiv in rand_pop:
        assert np.all(indiv <= max_val)
        assert np.all(indiv >= min_val)
        assert indiv.size == 10


def test_uniform_int_vec_init():
    pop_init = UniformVectorInitializer(200, 0, 1, 100, dtype=int)
    rand_inidv = pop_init.generate_random()
    assert 0 in rand_inidv
    assert 1 in rand_inidv


def test_uniform_err_vec_init():
    with pytest.raises(ValueError):
        UniformVectorInitializer(10, np.zeros(10), np.ones(9))

    with pytest.raises(ValueError):
        UniformVectorInitializer(10, np.zeros(9), np.ones(10))


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
        rand_inidv = pop_init.generate_random()
        assert rand_inidv.size == vec_size

        rand_inidv = pop_init.generate_individual()
        assert rand_inidv.size == vec_size

    rand_pop = pop_init.generate_population(None)
    assert len(rand_pop) == pop_size

    for indiv in rand_pop:
        assert indiv.size == vec_size


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
        rand_inidv = pop_init.generate_random()
        assert rand_inidv.size == 10

        rand_inidv = pop_init.generate_individual()
        assert rand_inidv.size == 10

    rand_pop = pop_init.generate_population(None)
    assert len(rand_pop) == pop_size

    for indiv in rand_pop:
        assert indiv.size == 10


def test_gaussian_int_vec_init():
    pop_init = GaussianVectorInitializer(200, 0, 1, 100, dtype=int)
    rand_inidv = pop_init.generate_random()


def test_gaussian_err_vec_init():
    with pytest.raises(ValueError):
        GaussianVectorInitializer(10, np.zeros(10), np.ones(9))

    with pytest.raises(ValueError):
        GaussianVectorInitializer(10, np.zeros(9), np.ones(10))


# @pytest.mark.parametrize("population", [example_populaton1, example_populaton2, example_populaton3])
# def test_direct_initializer(population):
#     print(population[0])
#     default_pop_init = GaussianVectorInitializer(population[0].genotype.size, -100, 100)
#     pop_init = DirectInitializer(default_pop_init, population)

#     # ids_in_pop = [i.id for i in population]

#     for _ in range(30):
#         rand_inidv = pop_init.generate_random()
#         # assert rand_inidv.id not in ids_in_pop

#         rand_inidv = pop_init.generate_individual()
#         # assert rand_inidv.id in ids_in_pop

#     rand_pop = pop_init.generate_population(None)
#     assert len(rand_pop) == pop_size

#     # for indiv in rand_pop:
#     #     assert indiv.id in ids_in_pop


# @pytest.mark.parametrize("population", [example_populaton1, example_populaton2, example_populaton3])
# def test_seed_prob_initializer(population):
#     default_pop_init = GaussianVectorInitializer(population[0].genotype.size, -100, 100)
#     pop_init = SeedProbInitializer(default_pop_init, population)

#     ids_in_pop = [i.id for i in population]

#     for _ in range(30):
#         rand_inidv = pop_init.generate_random()
#         # assert rand_inidv.id not in ids_in_pop

#         rand_inidv = pop_init.generate_individual()

#     rand_pop = pop_init.generate_population(None)


# @pytest.mark.parametrize("population", [example_populaton1, example_populaton2, example_populaton3])
# def test_seed_determ_initializer(population):
#     default_pop_init = GaussianVectorInitializer(population[0].genotype.size, -100, 100)
#     pop_init = SeedDetermInitializer(default_pop_init, population, 4)

#     # ids_in_pop = [i.id for i in population]

#     for _ in range(30):
#         rand_inidv = pop_init.generate_random()
#         # assert rand_inidv.id not in ids_in_pop

#         rand_inidv = pop_init.generate_individual()

#     rand_pop = pop_init.generate_population(None)


# @pytest.mark.parametrize("population", [example_populaton1, example_populaton2, example_populaton3])
# def test_seed_determ_null_param_initializer(population):
#     default_pop_init = GaussianVectorInitializer(population[0].genotype.size, -100, 100)
#     pop_init = SeedDetermInitializer(default_pop_init, population)


# @pytest.mark.parametrize("vec_size", [1, 2, 10, 100])
# @pytest.mark.parametrize("pop_size", [1, 10, 100])
# def test_lambda_init(vec_size, pop_size):
#     generator = lambda: np.zeros(vec_size)
#     pop_init = InitializerFromLambda(generator, pop_size)

#     for _ in range(30):
#         rand_inidv = pop_init.generate_random()
#         assert rand_inidv.size == vec_size

#         rand_inidv = pop_init.generate_individual()
#         assert rand_inidv.size == vec_size

#     rand_pop = pop_init.generate_population(None)
#     assert len(rand_pop) == pop_size

#     for indiv in rand_pop:
#         assert indiv.size == vec_size


@pytest.mark.parametrize("vec_size", [2, 10, 100])
@pytest.mark.parametrize("pop_size", [1, 10, 100])
def test_perm_init(vec_size, pop_size):
    pop_init = PermInitializer(vec_size, pop_size)

    for _ in range(30):
        rand_inidv = pop_init.generate_random()
        assert np.all(np.isin(np.arange(vec_size), rand_inidv))

        rand_inidv = pop_init.generate_individual()
        assert np.all(np.isin(np.arange(vec_size), rand_inidv))

    rand_pop = pop_init.generate_population(None)
    assert len(rand_pop) == pop_size

    for indiv in rand_pop:
        assert np.all(np.isin(np.arange(vec_size), rand_inidv))
