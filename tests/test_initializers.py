import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from conftest import (
    DummyObjectiveFunction,
    DummyParameterExtendingEncoding,
    _expected_exponential,
    _expected_normal,
    _expected_uniform,
    _expected_permutation,
    rng,
    dummy_objfunc,
    simple_encoding,
)

from metaheuristic_designer.initializers import (
    ExponentialInitializer,
    GaussianInitializer,
    UniformInitializer,
    PermInitializer,
    SeedProbInitializer,
    SeedDetermInitializer,
    DirectInitializer,
    ExtendedInitializer,
)
from metaheuristic_designer.initializer import InitializerFromLambda
from metaheuristic_designer.population import Population


# ===================================================================
#  ExponentialInitializer
# ===================================================================
@pytest.mark.parametrize(
    "genotype_size, beta, dtype, pop_size",
    [
        (3, 1.0, float, 1),
        (5, 2.0, float, 4),
        (4, 0.5, int, 1),
        (3, 1.5, int, 2),
        (1, 10.0, float, 1),
    ],
)
def test_exponential_generate_random_shape_and_type(genotype_size, beta, dtype, pop_size, rng):
    init = ExponentialInitializer(genotype_size, beta, pop_size=pop_size, dtype=dtype, random_state=rng)
    vec = init.generate_random()
    assert vec.shape == (genotype_size,)
    assert vec.dtype == np.dtype(dtype)
    assert np.all(vec >= 0)


@pytest.mark.parametrize(
    "genotype_size, beta, dtype",
    [
        (3, 1.0, float),
        (2, 2.5, int),
    ],
)
def test_exponential_generate_random_deterministic(genotype_size, beta, dtype, rng):
    init = ExponentialInitializer(genotype_size, beta, pop_size=1, dtype=dtype, random_state=rng)
    rng_expected = np.random.default_rng(42)
    expected_init = ExponentialInitializer(genotype_size, beta, pop_size=1, dtype=dtype, random_state=rng_expected)
    expected = expected_init.generate_random()
    assert_array_equal(init.generate_random(), expected)


def test_exponential_generate_population(rng, dummy_objfunc, simple_encoding):
    init = ExponentialInitializer(2, 1.0, pop_size=4, encoding=simple_encoding, random_state=rng)
    pop = init.generate_population(dummy_objfunc)
    assert len(pop) == 4
    assert pop.genotype_matrix.shape == (4, 2)
    assert pop.objfunc is dummy_objfunc
    assert pop.encoding is simple_encoding


# ===================================================================
#  GaussianInitializer
# ===================================================================
@pytest.mark.parametrize(
    "genotype_size, g_mean, g_std, dtype, pop_size",
    [
        (3, 0.0, 1.0, float, 1),
        (4, 5.0, 0.5, float, 2),
        (2, 10.0, 0.1, int, 1),
        (1, 0.0, 2.0, float, 1),
    ],
)
def test_gaussian_generate_random_shape_and_type(genotype_size, g_mean, g_std, dtype, pop_size, rng):
    init = GaussianInitializer(genotype_size, g_mean, g_std, pop_size=pop_size, dtype=dtype, random_state=rng)
    vec = init.generate_random()
    assert vec.shape == (genotype_size,)
    assert vec.dtype == np.dtype(dtype)


@pytest.mark.parametrize(
    "genotype_size, g_mean, g_std, dtype",
    [
        (2, 0.0, 1.0, float),
        (3, 2.0, 0.5, int),
    ],
)
def test_gaussian_generate_random_deterministic(genotype_size, g_mean, g_std, dtype, rng):
    init = GaussianInitializer(genotype_size, g_mean, g_std, pop_size=1, dtype=dtype, random_state=rng)
    rng_expected = np.random.default_rng(42)
    expected_init = GaussianInitializer(genotype_size, g_mean, g_std, pop_size=1, dtype=dtype, random_state=rng_expected)
    expected = expected_init.generate_random()
    assert_array_equal(init.generate_random(), expected)


def test_gaussian_sequence_parameters(rng):
    GaussianInitializer(3, [1, 2, 3], [0.1, 0.2, 0.3], random_state=rng)
    with pytest.raises(ValueError):
        GaussianInitializer(3, [1, 2], [0.1, 0.2, 0.3], random_state=rng)
    with pytest.raises(ValueError):
        GaussianInitializer(3, [1, 2, 3], [0.1, 0.2], random_state=rng)


def test_gaussian_generate_population(rng, dummy_objfunc):
    init = GaussianInitializer(2, 1.0, 0.2, pop_size=5, random_state=rng)
    pop = init.generate_population(dummy_objfunc)
    assert len(pop) == 5
    assert pop.genotype_matrix.shape == (5, 2)


# ===================================================================
#  UniformInitializer
# ===================================================================
@pytest.mark.parametrize(
    "genotype_size, low, high, dtype, pop_size",
    [
        (3, -1.0, 1.0, float, 1),
        (4, 0.0, 10.0, float, 3),
        (2, 2.0, 5.0, int, 1),
        (1, 0.0, 100.0, float, 1),
    ],
)
def test_uniform_generate_random_shape_and_type(genotype_size, low, high, dtype, pop_size, rng):
    init = UniformInitializer(genotype_size, low, high, pop_size=pop_size, dtype=dtype, random_state=rng)
    vec = init.generate_random()
    assert vec.shape == (genotype_size,)
    assert vec.dtype == np.dtype(dtype)
    assert np.all(vec >= low)
    assert np.all(vec <= high)


@pytest.mark.parametrize(
    "genotype_size, low, high, dtype",
    [
        (2, 0.0, 1.0, float),
        (3, 5.0, 10.0, int),
    ],
)
def test_uniform_generate_random_deterministic(genotype_size, low, high, dtype, rng):
    init = UniformInitializer(genotype_size, low, high, pop_size=1, dtype=dtype, random_state=rng)
    rng_expected = np.random.default_rng(42)
    expected_init = UniformInitializer(genotype_size, low, high, pop_size=1, dtype=dtype, random_state=rng_expected)
    expected = expected_init.generate_random()
    assert_array_equal(init.generate_random(), expected)


def test_uniform_sequence_parameters(rng):
    UniformInitializer(3, [0, 0, 0], [1, 2, 3], random_state=rng)
    with pytest.raises(ValueError):
        UniformInitializer(3, [0, 0], [1, 2, 3], random_state=rng)
    with pytest.raises(ValueError):
        UniformInitializer(3, [0, 0, 0], [1, 2], random_state=rng)


def test_uniform_generate_population(rng, dummy_objfunc):
    init = UniformInitializer(2, -1, 1, pop_size=4, random_state=rng)
    pop = init.generate_population(dummy_objfunc)
    assert len(pop) == 4
    assert pop.genotype_matrix.shape == (4, 2)


# ===================================================================
#  PermInitializer
# ===================================================================
@pytest.mark.parametrize("n", [1, 5, 10])
def test_perm_generate_random(n, rng):
    init = PermInitializer(n, random_state=rng)
    perm = init.generate_random()
    assert perm.shape == (n,)
    np.testing.assert_array_equal(np.sort(perm), np.arange(n))


def test_perm_generate_random_deterministic(rng):
    init = PermInitializer(5, random_state=rng)
    rng_expected = np.random.default_rng(42)
    expected_init = PermInitializer(5, random_state=rng_expected)
    expected = expected_init.generate_random()
    assert_array_equal(init.generate_random(), expected)


def test_perm_generate_population(rng, dummy_objfunc):
    init = PermInitializer(3, pop_size=5, random_state=rng)
    pop = init.generate_population(dummy_objfunc)
    assert len(pop) == 5
    assert pop.genotype_matrix.shape == (5, 3)
    for row in pop.genotype_matrix:
        np.testing.assert_array_equal(np.sort(row), np.arange(3))


# ===================================================================
#  SeedProbInitializer
# ===================================================================
def test_seed_prob_generate_random_returns_valid_shape(rng):
    default_init = UniformInitializer(2, 0, 1, random_state=rng)
    seed_init = SeedProbInitializer(default_init, solutions=np.array([[9, 9]]), insert_prob=0.0, random_state=rng)
    vec = seed_init.generate_random()
    assert vec.shape == (2,)
    assert np.all(vec >= 0) and np.all(vec <= 1)


@pytest.mark.parametrize("insert_prob", [0.0, 1.0])
def test_seed_prob_individual_insertion(insert_prob, rng):
    # Use identical seeds for two separate initializers
    rng_default = np.random.default_rng(42)
    rng_seed = np.random.default_rng(42)
    default_init = UniformInitializer(2, 0, 1, random_state=rng_default)
    solutions = np.array([[42.0, 42.0]])
    seed_init = SeedProbInitializer(default_init, solutions=solutions, insert_prob=insert_prob, random_state=rng_seed)
    indiv = seed_init.generate_individual()
    if insert_prob == 0.0:
        # Must be a uniform draw, not the seed
        expected_init = UniformInitializer(2, 0, 1, random_state=np.random.default_rng(42))
        assert_array_equal(indiv, expected_init.generate_random())
    else:
        assert_array_equal(indiv, [42.0, 42.0])


def test_seed_prob_generate_population_mixed(rng, dummy_objfunc):
    solutions = np.array([[100, 100]])
    default_init = UniformInitializer(2, 0, 1, random_state=rng)
    seed_init = SeedProbInitializer(default_init, solutions=solutions, insert_prob=0.5, random_state=np.random.default_rng(43))
    pop = seed_init.generate_population(dummy_objfunc, n_individuals=10)
    assert len(pop) == 10
    assert pop.genotype_matrix.shape == (10, 2)


# ===================================================================
#  SeedDetermInitializer
# ===================================================================
def test_seed_determ_inserts_exact_number(rng, dummy_objfunc):
    solutions = np.array([[10, 20], [30, 40]])
    default_init = UniformInitializer(2, 0, 1, random_state=rng)
    init = SeedDetermInitializer(default_init, solutions=solutions, n_to_insert=2, random_state=rng)
    pop = init.generate_population(dummy_objfunc, n_individuals=5)
    assert_array_equal(pop.genotype_matrix[0], [10, 20])
    assert_array_equal(pop.genotype_matrix[1], [30, 40])
    assert len(pop) == 5
    pop2 = init.generate_population(dummy_objfunc, n_individuals=5)
    assert_array_equal(pop2.genotype_matrix[0], [10, 20])


def test_seed_determ_wraps_around_seed_list(rng, dummy_objfunc):
    solutions = np.array([[1, 1]])
    default_init = UniformInitializer(2, 0, 1, random_state=rng)
    init = SeedDetermInitializer(default_init, solutions=solutions, n_to_insert=3, random_state=rng)
    pop = init.generate_population(dummy_objfunc, n_individuals=3)
    expected = np.tile([1, 1], (3, 1))
    assert_array_equal(pop.genotype_matrix, expected)


def test_seed_determ_no_insert_inserts_default(rng, dummy_objfunc):
    solutions = np.array([[7, 7]])
    default_init = UniformInitializer(2, 0, 1, random_state=rng)
    init = SeedDetermInitializer(default_init, solutions=solutions, n_to_insert=0, random_state=rng)
    pop = init.generate_population(dummy_objfunc, n_individuals=4)
    assert not np.any(np.all(pop.genotype_matrix == [7, 7], axis=1))


# ===================================================================
#  DirectInitializer
# ===================================================================
def test_direct_individual_from_array(rng):
    solutions = np.array([[5, 6], [7, 8]])
    default_init = UniformInitializer(2, 0, 1, random_state=rng)
    init = DirectInitializer(default_init, solutions, random_state=rng)
    indiv = init.generate_individual()
    assert indiv.shape == (2,)
    assert any(np.array_equal(indiv, row) for row in solutions)


def test_direct_individual_from_population(rng, dummy_objfunc):
    solutions = np.array([[1, 2], [3, 4]])
    pop = Population(dummy_objfunc, solutions)
    default_init = UniformInitializer(2, 0, 1, random_state=rng)
    init = DirectInitializer(default_init, pop, random_state=rng)
    indiv = init.generate_individual()
    assert any(np.array_equal(indiv, row) for row in solutions)


def test_direct_generate_population_from_array(rng, dummy_objfunc):
    solutions = np.array([[10, 20], [30, 40], [50, 60]])
    default_init = UniformInitializer(2, 0, 1, random_state=rng)
    init = DirectInitializer(default_init, solutions, random_state=rng)
    pop = init.generate_population(dummy_objfunc, n_individuals=5)
    assert len(pop) == 5
    assert_array_equal(pop.genotype_matrix[0], [10, 20])
    assert_array_equal(pop.genotype_matrix[1], [30, 40])
    assert_array_equal(pop.genotype_matrix[2], [50, 60])
    assert_array_equal(pop.genotype_matrix[3], [10, 20])
    assert_array_equal(pop.genotype_matrix[4], [30, 40])


def test_direct_generate_population_from_population_exact(rng, dummy_objfunc):
    solutions = np.array([[1, 1], [2, 2]])
    pop_in = Population(dummy_objfunc, solutions)
    pop_in.fitness = np.array([10.0, 20.0])
    default_init = UniformInitializer(2, 0, 1, random_state=rng)
    init = DirectInitializer(default_init, pop_in, random_state=rng)
    pop_out = init.generate_population(dummy_objfunc, n_individuals=2)
    assert len(pop_out) == 2
    assert_array_equal(pop_out.genotype_matrix, solutions)
    assert_array_equal(pop_out.fitness, pop_in.fitness)


# ===================================================================
#  ExtendedInitializer
# ===================================================================
def test_extended_generate_random(rng):
    encoding = DummyParameterExtendingEncoding([("a", 2), ("b", 1)])
    solution_init = UniformInitializer(3, 0, 1, random_state=rng)
    param_inits = {
        "a": UniformInitializer(2, 0, 1, random_state=np.random.default_rng(100)),
        "b": UniformInitializer(1, 0, 1, random_state=np.random.default_rng(101)),
    }
    init = ExtendedInitializer(solution_init, param_inits, encoding, random_state=rng)

    # Build a second identical instance for reproducibility check
    rng_expected = np.random.default_rng(42)
    solution_init2 = UniformInitializer(3, 0, 1, random_state=np.random.default_rng(42))
    param_inits2 = {
        "a": UniformInitializer(2, 0, 1, random_state=np.random.default_rng(100)),
        "b": UniformInitializer(1, 0, 1, random_state=np.random.default_rng(101)),
    }
    init2 = ExtendedInitializer(solution_init2, param_inits2, encoding, random_state=rng_expected)

    vec = init.generate_random()
    expected = init2.generate_random()
    assert_array_equal(vec, expected)
    assert vec.shape == (6,)


def test_extended_generate_individual_same_structure(rng):
    encoding = DummyParameterExtendingEncoding([("c", 2)])
    solution_init = GaussianInitializer(2, 0, 1, random_state=rng)
    param_inits = {"c": GaussianInitializer(2, 5, 0.1, random_state=np.random.default_rng(200))}
    init = ExtendedInitializer(solution_init, param_inits, encoding, random_state=rng)

    # Second identical instance
    rng_expected = np.random.default_rng(42)
    solution_init2 = GaussianInitializer(2, 0, 1, random_state=np.random.default_rng(42))
    param_inits2 = {"c": GaussianInitializer(2, 5, 0.1, random_state=np.random.default_rng(200))}
    init2 = ExtendedInitializer(solution_init2, param_inits2, encoding, random_state=rng_expected)

    indiv = init.generate_individual()
    expected = init2.generate_individual()
    assert_array_equal(indiv, expected)
    assert indiv.shape == (4,)


# ===================================================================
#  InitializerFromLambda
# ===================================================================
def test_lambda_generate_random(rng):
    def my_gen(rs):
        return rs.uniform(10, 20, size=3)

    init = InitializerFromLambda(my_gen, pop_size=2, random_state=rng)
    vec = init.generate_random()
    assert vec.shape == (3,)
    assert np.all(vec >= 10) and np.all(vec <= 20)


def test_lambda_generate_individual_calls_same(rng):
    def my_gen(rs):
        return np.array([1.0, 2.0])

    init = InitializerFromLambda(my_gen, random_state=rng)
    assert_array_equal(init.generate_individual(), np.array([1.0, 2.0]))


def test_lambda_generate_population(rng, dummy_objfunc):
    def my_gen(rs):
        return rs.integers(0, 100, size=2)

    init = InitializerFromLambda(my_gen, pop_size=5, random_state=rng)
    pop = init.generate_population(dummy_objfunc)
    assert len(pop) == 5
    assert pop.genotype_matrix.shape == (5, 2)
    assert np.issubdtype(pop.genotype_matrix.dtype, np.integer)
    assert np.all(pop.genotype_matrix >= 0)
    assert np.all(pop.genotype_matrix < 100)


# -------------------------------------------------------------------
#  Reproducibility suite
# -------------------------------------------------------------------
@pytest.mark.parametrize(
    "init_factory",
    [
        lambda rng: ExponentialInitializer(3, 1.0, random_state=rng),
        lambda rng: GaussianInitializer(3, 0.0, 1.0, random_state=rng),
        lambda rng: UniformInitializer(3, -1.0, 1.0, random_state=rng),
        lambda rng: PermInitializer(5, random_state=rng),
        lambda rng: SeedProbInitializer(
            UniformInitializer(3, -1.0, 1.0, random_state=np.random.default_rng(999)),
            solutions=np.array([[9, 9, 9]]),
            insert_prob=0.5,
            random_state=rng,
        ),
        lambda rng: DirectInitializer(
            UniformInitializer(3, -1.0, 1.0, random_state=np.random.default_rng(888)),
            solutions=np.array([[1, 2, 3], [4, 5, 6]]),
            random_state=rng,
        ),
        lambda rng: ExtendedInitializer(
            UniformInitializer(3, 0, 1, random_state=rng),
            {"sigma": GaussianInitializer(1, 0, 1, random_state=np.random.default_rng(777))},
            DummyParameterExtendingEncoding([("sigma", 1)]),
            random_state=rng,
        ),
        lambda rng: InitializerFromLambda(
            lambda rs: rs.uniform(10, 20, size=3),
            random_state=rng,
        ),
    ],
)
def test_reproducible_initializers(init_factory, rng):
    init1 = init_factory(rng)
    init2 = init_factory(np.random.default_rng(42))
    for _ in range(5):
        v1 = init1.generate_random()
        v2 = init2.generate_random()
        assert_array_equal(v1, v2)


def test_generate_individual_matches_generate_random_with_same_seed(rng):
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    init_a = GaussianInitializer(3, 0, 1, random_state=rng_a)
    init_b = GaussianInitializer(3, 0, 1, random_state=rng_b)
    assert_array_equal(init_a.generate_individual(), init_b.generate_random())
