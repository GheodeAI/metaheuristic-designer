import pytest
import numpy as np
from metaheuristic_designer import InitializerFromLambda
from metaheuristic_designer.initializers import *
from metaheuristic_designer.encodings import ParameterExtendingEncoding


@pytest.mark.parametrize("vec_size", [1, 2, 10, 100])
@pytest.mark.parametrize("min_val, max_val", [(0, 1), (-1, 1), (-100, 2), (2, 24)])
@pytest.mark.parametrize("pop_size", [1, 10, 100])
def test_uniform_vec_init(vec_size, min_val, max_val, pop_size):
    pop_init = UniformInitializer(vec_size, min_val, max_val, pop_size)

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
    pop_init = UniformInitializer(10, min_val, max_val, pop_size)

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
    pop_init = UniformInitializer(200, 0, 1, 100, dtype=int)
    rand_inidv = pop_init.generate_random()
    assert 0 in rand_inidv
    assert 1 in rand_inidv


def test_uniform_err_vec_init():
    with pytest.raises(ValueError):
        UniformInitializer(10, np.zeros(10), np.ones(9))

    with pytest.raises(ValueError):
        UniformInitializer(10, np.zeros(9), np.ones(10))


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
    pop_init = GaussianInitializer(vec_size, mean_val, std_val, pop_size)

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
    pop_init = GaussianInitializer(10, min_val, max_val, pop_size)

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
    pop_init = GaussianInitializer(200, 0, 1, 100, dtype=int)
    rand_inidv = pop_init.generate_random()


def test_gaussian_err_vec_init():
    with pytest.raises(ValueError):
        GaussianInitializer(10, np.zeros(10), np.ones(9))

    with pytest.raises(ValueError):
        GaussianInitializer(10, np.zeros(9), np.ones(10))


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
        assert np.all(np.isin(np.arange(vec_size), indiv))


@pytest.mark.parametrize("pop_size", [1, 10, 100])
@pytest.mark.parametrize("vec_size", [1, 10, 100])
def test_initialize_lambda(pop_size, vec_size):
    pop_init = InitializerFromLambda(lambda _: np.zeros(vec_size), pop_size)

    rand_inidv = pop_init.generate_random()
    assert rand_inidv.shape[0] == vec_size
    assert np.all(rand_inidv == 0)

    rand_inidv = pop_init.generate_random()
    assert rand_inidv.shape[0] == vec_size
    assert np.all(rand_inidv == 0)

    rand_population_matrix = pop_init.generate_population(None).genotype_matrix
    assert rand_population_matrix.shape == (pop_size, vec_size)
    assert np.all(rand_population_matrix == 0)


@pytest.mark.parametrize("vec_size", [1, 2, 10, 100])
@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("pop_size", [1, 10, 100])
def test_exponential_vec_init(vec_size, beta, pop_size):
    pop_init = ExponentialInitializer(vec_size, beta, pop_size)

    for _ in range(30):
        for method in [pop_init.generate_random, pop_init.generate_individual]:
            indiv = method()
            assert indiv.shape == (vec_size,)
            # exponential distribution produces non‑negative values
            assert indiv.min() >= 0.0

    rand_pop = pop_init.generate_population(None)
    assert len(rand_pop) == pop_size
    assert rand_pop.genotype_matrix.shape == (pop_size, vec_size)
    # All individuals from population should also be non‑negative
    for indiv in rand_pop:
        assert indiv.min() >= 0.0


def test_exponential_int_dtype():
    pop_init = ExponentialInitializer(10, 0.5, 100, dtype=int)
    indiv = pop_init.generate_random()
    assert indiv.shape == (10,)
    assert indiv.dtype == int


def test_direct_initializer_uses_provided_solutions():
    # Create a default initializer that always produces zeros
    default_init = InitializerFromLambda(lambda _: np.zeros(5), pop_size=1)
    # Provide a few distinct solutions
    sol_list = np.vstack([np.arange(5), np.arange(5) + 10, np.arange(5) + 20])
    direct_init = DirectInitializer(default_init, sol_list)

    for _ in range(30):
        rand = direct_init.generate_random()
        print(rand)
        assert all(rand == 0)

    # generate_individual likely returns a solution as well
    indiv = direct_init.generate_individual()
    assert any(np.array_equal(indiv, s) for s in sol_list)

    # generate_population: we request a population of size 10
    pop = direct_init.generate_population(None, n_individuals=10)
    assert len(pop) == 10
    # Some individuals should be from the seed list, others from default (zeros)
    # We can't guarantee exact distribution, but at least all shapes are (5,)
    for ind in pop:
        assert ind.shape == (5,)


# ============================================================================
# SeedProbInitializer
# ============================================================================
def test_seed_prob_initializer_shape():
    default_init = InitializerFromLambda(lambda _: np.ones(3), pop_size=1)
    seeds = [np.zeros(3), np.full(3, 2.0)]
    seed_init = SeedProbInitializer(default_init, seeds, insert_prob=0.3)

    for _ in range(20):
        indiv = seed_init.generate_random()
        assert indiv.shape == (3,)

    pop = seed_init.generate_population(None, n_individuals=8)
    assert len(pop) == 8
    for ind in pop:
        assert ind.shape == (3,)


# ============================================================================
# SeedDetermInitializer
# ============================================================================
def test_seed_determ_initializer_shape():
    default_init = InitializerFromLambda(lambda _: np.ones(3), pop_size=1)
    seeds = [np.zeros(3), np.full(3, -1.0)]
    seed_init = SeedDetermInitializer(default_init, seeds, n_to_insert=2)

    for _ in range(20):
        indiv = seed_init.generate_random()
        assert indiv.shape == (3,)

    pop = seed_init.generate_population(None, n_individuals=6)
    assert len(pop) == 6
    for ind in pop:
        assert ind.shape == (3,)


# ============================================================================
# ExtendedInitializer (requires a ParameterExtendingEncoding)
# ============================================================================
def test_extended_initializer_concatenates():
    # Simple encoding: 5 solution genes + 3 "speed" parameter genes
    encoding = ParameterExtendingEncoding(
        vecsize=5,
        param_sizes=[("speed", 3)],
        base_encoding=None
    )
    solution_init = UniformInitializer(5, -1, 1)
    param_init_dict = {"speed": GaussianInitializer(3, 0, 1)}
    ext_init = ExtendedInitializer(
        solution_init, param_init_dict, encoding
    )

    indiv = ext_init.generate_random()
    # Total length = 5 (solution) + 3 (speed) = 8
    assert indiv.shape == (8,)

    indiv2 = ext_init.generate_individual()
    assert indiv2.shape == (8,)

    pop = ext_init.generate_population(None)
    assert len(pop) == 20
    assert pop.genotype_matrix.shape == (20, 8)

# ============================================================================
# Reproducibility tests
# ============================================================================

def test_uniform_reproducible():
    seed = 42
    p1 = UniformInitializer(10, -5, 5, 100, random_state=seed)
    p2 = UniformInitializer(10, -5, 5, 100, random_state=seed)
    for _ in range(5):
        assert np.array_equal(p1.generate_random(), p2.generate_random())
        assert np.array_equal(p1.generate_individual(), p2.generate_individual())

    # different seeds should (almost certainly) give different values
    p3 = UniformInitializer(10, -5, 5, 100, random_state=seed+1)
    assert not np.array_equal(p1.generate_random(), p3.generate_random())


def test_gaussian_reproducible():
    seed = 123
    p1 = GaussianInitializer(10, 0, 1, 50, random_state=seed)
    p2 = GaussianInitializer(10, 0, 1, 50, random_state=seed)
    for _ in range(5):
        assert np.array_equal(p1.generate_random(), p2.generate_random())
        assert np.array_equal(p1.generate_individual(), p2.generate_individual())

    p3 = GaussianInitializer(10, 0, 1, 50, random_state=seed+1)
    assert not np.array_equal(p1.generate_random(), p3.generate_random())


def test_exponential_reproducible():
    seed = 7
    p1 = ExponentialInitializer(10, 0.5, 30, random_state=seed)
    p2 = ExponentialInitializer(10, 0.5, 30, random_state=seed)
    for _ in range(5):
        assert np.array_equal(p1.generate_random(), p2.generate_random())
        assert np.array_equal(p1.generate_individual(), p2.generate_individual())

    p3 = ExponentialInitializer(10, 0.5, 30, random_state=seed+1)
    assert not np.array_equal(p1.generate_random(), p3.generate_random())


def test_perm_reproducible():
    seed = 99
    p1 = PermInitializer(10, 20, random_state=seed)
    p2 = PermInitializer(10, 20, random_state=seed)
    for _ in range(5):
        assert np.array_equal(p1.generate_random(), p2.generate_random())
        assert np.array_equal(p1.generate_individual(), p2.generate_individual())

    p3 = PermInitializer(10, 20, random_state=seed+1)
    assert not np.array_equal(p1.generate_random(), p3.generate_random())


def test_lambda_init_reproducible():
    # Lambda that uses the random state (but returns zeros regardless)
    p1 = InitializerFromLambda(lambda rng: np.zeros(5), pop_size=100, random_state=42)
    p2 = InitializerFromLambda(lambda rng: np.zeros(5), pop_size=100, random_state=42)
    for _ in range(5):
        assert np.array_equal(p1.generate_random(), p2.generate_random())
        assert np.array_equal(p1.generate_individual(), p2.generate_individual())
    # with different seeds the output is still zero, so no difference test needed


def test_direct_init_reproducible():
    seed = 11
    seeds = np.vstack([np.arange(5), np.full(5, -1)])
    p1 = DirectInitializer(UniformInitializer(5, -1, 1, 1, random_state=seed), seeds, random_state=seed)
    p2 = DirectInitializer(UniformInitializer(5, -1, 1, 1, random_state=seed), seeds, random_state=seed)
    for _ in range(10):
        assert np.array_equal(p1.generate_random(), p2.generate_random())
        assert np.array_equal(p1.generate_individual(), p2.generate_individual())


def test_seed_prob_reproducible():
    seed = 22
    seeds = np.vstack([np.zeros(3), np.ones(3)])
    p1 = SeedProbInitializer(UniformInitializer(3, 0, 1, 1, random_state=seed), seeds, insert_prob=0.5, random_state=seed)
    p2 = SeedProbInitializer(UniformInitializer(3, 0, 1, 1, random_state=seed), seeds, insert_prob=0.5, random_state=seed)
    for _ in range(10):
        assert np.array_equal(p1.generate_random(), p2.generate_random())
        assert np.array_equal(p1.generate_individual(), p2.generate_individual())


def test_seed_determ_reproducible():
    seed = 33
    seeds = np.vstack([np.zeros(3), np.ones(3)])
    p1 = SeedDetermInitializer(UniformInitializer(3, 0, 1, 1, random_state=seed), seeds, n_to_insert=2, random_state=seed)
    p2 = SeedDetermInitializer(UniformInitializer(3, 0, 1, 1, random_state=seed), seeds, n_to_insert=2, random_state=seed)
    for _ in range(10):
        assert np.array_equal(p1.generate_random(), p2.generate_random())
        assert np.array_equal(p1.generate_individual(), p2.generate_individual())


def test_extended_init_reproducible():
    seed = 44
    encoding = ParameterExtendingEncoding(
        vecsize=5, param_sizes=[("speed", 3)], base_encoding=None
    )
    param_init = {"speed": GaussianInitializer(3, 0, 1, random_state=seed+1)}
    p1 = ExtendedInitializer(UniformInitializer(5, -1, 1, random_state=seed), param_init, encoding, pop_size=20, random_state=seed)
    p2 = ExtendedInitializer(UniformInitializer(5, -1, 1, random_state=seed), param_init, encoding, pop_size=20, random_state=seed)
    for _ in range(5):
        assert np.array_equal(p1.generate_random(), p2.generate_random())
        assert np.array_equal(p1.generate_individual(), p2.generate_individual())