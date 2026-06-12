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
    SeededInitializer,
    FixedSeededInitializer,
    CompositeInitializer,
    FixedCompositeInitializer,
    DirectInitializer,
    ExtendedInitializer,
    LatinHypercubeInitializer,
    SobolInitializer,
    HaltonInitializer,
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
    init = ExponentialInitializer(genotype_size, beta, pop_size=pop_size, dtype=dtype, rng=rng)
    vec = init.generate_random()
    assert vec.shape == (genotype_size,)
    assert vec.dtype == np.dtype(dtype)
    assert np.all(vec >= 0)


def test_exponential_generate_population(rng, simple_encoding):
    init = ExponentialInitializer(2, 1.0, pop_size=4, encoding=simple_encoding, rng=rng)
    pop = init.generate_population()
    assert len(pop) == 4
    assert pop.genotype_matrix.shape == (4, 2)
    assert pop.encoding is simple_encoding


def test_exponential_reproducible_random():
    init1 = ExponentialInitializer(2, 1.0, pop_size=4, rng=42)
    init2 = ExponentialInitializer(2, 1.0, pop_size=4, rng=42)
    for _ in range(5):
        v1 = init1.generate_random()
        v2 = init2.generate_random()
        assert_array_equal(v1, v2)


def test_exponential_reproducible_population():
    init1 = ExponentialInitializer(2, 1.0, pop_size=4, rng=42)
    init2 = ExponentialInitializer(2, 1.0, pop_size=4, rng=42)
    for _ in range(5):
        v1 = init1.generate_population()
        v2 = init2.generate_population()
        assert_array_equal(v1.genotype_matrix, v2.genotype_matrix)


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
    init = GaussianInitializer(genotype_size, g_mean, g_std, pop_size=pop_size, dtype=dtype, rng=rng)
    vec = init.generate_random()
    assert vec.shape == (genotype_size,)
    assert vec.dtype == np.dtype(dtype)


def test_gaussian_sequence_parameters(rng):
    GaussianInitializer(3, [1, 2, 3], [0.1, 0.2, 0.3], rng=rng)
    with pytest.raises(ValueError):
        GaussianInitializer(3, [1, 2], [0.1, 0.2, 0.3], rng=rng)
    with pytest.raises(ValueError):
        GaussianInitializer(3, [1, 2, 3], [0.1, 0.2], rng=rng)


def test_gaussian_generate_population(rng):
    init = GaussianInitializer(2, 1.0, 0.2, pop_size=5, rng=rng)
    pop = init.generate_population()
    assert len(pop) == 5
    assert pop.genotype_matrix.shape == (5, 2)


def test_gaussian_reproducible_random():
    init1 = GaussianInitializer(2, 1.0, 0.2, pop_size=5, rng=42)
    init2 = GaussianInitializer(2, 1.0, 0.2, pop_size=5, rng=42)
    for _ in range(5):
        v1 = init1.generate_random()
        v2 = init2.generate_random()
        assert_array_equal(v1, v2)


def test_gaussian_reproducible_population():
    init1 = GaussianInitializer(2, 1.0, 0.2, pop_size=5, rng=42)
    init2 = GaussianInitializer(2, 1.0, 0.2, pop_size=5, rng=42)
    for _ in range(5):
        v1 = init1.generate_population()
        v2 = init2.generate_population()
        assert_array_equal(v1.genotype_matrix, v2.genotype_matrix)


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
    init = UniformInitializer(genotype_size, low, high, population_size=pop_size, dtype=dtype, rng=rng)
    vec = init.generate_random()
    assert vec.shape == (genotype_size,)
    assert vec.dtype == np.dtype(dtype)
    assert np.all(vec >= low)
    assert np.all(vec <= high)


def test_uniform_sequence_parameters(rng):
    UniformInitializer(3, [0, 0, 0], [1, 2, 3], rng=rng)
    with pytest.raises(ValueError):
        UniformInitializer(3, [0, 0], [1, 2, 3], rng=rng)
    with pytest.raises(ValueError):
        UniformInitializer(3, [0, 0, 0], [1, 2], rng=rng)


def test_uniform_generate_population(rng):
    init = UniformInitializer(2, -1, 1, population_size=4, rng=rng)
    pop = init.generate_population()
    assert len(pop) == 4
    assert pop.genotype_matrix.shape == (4, 2)


def test_uniform_reproducible_random():
    init1 = UniformInitializer(2, -1, 1, population_size=4, rng=42)
    init2 = UniformInitializer(2, -1, 1, population_size=4, rng=42)
    for _ in range(5):
        v1 = init1.generate_random()
        v2 = init2.generate_random()
        assert_array_equal(v1, v2)


def test_uniform_reproducible_population():
    init1 = UniformInitializer(2, -1, 1, population_size=4, rng=42)
    init2 = UniformInitializer(2, -1, 1, population_size=4, rng=42)
    for _ in range(5):
        v1 = init1.generate_population()
        v2 = init2.generate_population()
        assert_array_equal(v1.genotype_matrix, v2.genotype_matrix)


# ===================================================================
#  PermInitializer
# ===================================================================
@pytest.mark.parametrize("dimension", [1, 5, 10])
@pytest.mark.parametrize("n", [1, 5, 10])
def test_perm_generate_random(dimension, n, rng):
    init = PermInitializer(dimension, n, rng=rng)
    perm = init.generate_random()
    assert perm.shape == (dimension,)
    assert_array_equal(np.sort(perm), np.arange(dimension))


def test_perm_generate_random_deterministic(rng):
    init = PermInitializer(5, rng=rng)
    rng_expected = np.random.default_rng(42)
    expected_init = PermInitializer(5, rng=rng_expected)
    expected = expected_init.generate_random()
    assert_array_equal(init.generate_random(), expected)


def test_perm_generate_population(rng):
    init = PermInitializer(3, population_size=5, rng=rng)
    pop = init.generate_population()
    assert len(pop) == 5
    assert pop.genotype_matrix.shape == (5, 3)
    for row in pop.genotype_matrix:
        assert_array_equal(np.sort(row), np.arange(3))


def test_perm_reproducible_random():
    init1 = PermInitializer(3, population_size=5, rng=42)
    init2 = PermInitializer(3, population_size=5, rng=42)
    for _ in range(5):
        v1 = init1.generate_random()
        v2 = init2.generate_random()
        assert_array_equal(v1, v2)


def test_perm_reproducible_population():
    init1 = PermInitializer(3, population_size=5, rng=42)
    init2 = PermInitializer(3, population_size=5, rng=42)
    for _ in range(5):
        v1 = init1.generate_population()
        v2 = init2.generate_population()
        assert_array_equal(v1.genotype_matrix, v2.genotype_matrix)


# ===================================================================
#  LatinHypercubeInitializer
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
def test_lhs_generate_random_shape_and_type(genotype_size, low, high, dtype, pop_size, rng):
    init = LatinHypercubeInitializer(genotype_size, low, high, population_size=pop_size, dtype=dtype, rng=rng)
    vec = init.generate_random()
    assert vec.shape == (genotype_size,)
    assert vec.dtype == np.dtype(dtype)
    assert np.all(vec >= low)
    assert np.all(vec <= high)


def test_lhs_sequence_parameters(rng):
    LatinHypercubeInitializer(3, [0, 0, 0], [1, 2, 3], rng=rng)
    with pytest.raises(ValueError):
        LatinHypercubeInitializer(3, [0, 0], [1, 2, 3], rng=rng)
    with pytest.raises(ValueError):
        LatinHypercubeInitializer(3, [0, 0, 0], [1, 2], rng=rng)


def test_lhs_generate_population(rng):
    init = LatinHypercubeInitializer(2, -1, 1, population_size=4, rng=rng)
    pop = init.generate_population()
    assert len(pop) == 4
    assert pop.genotype_matrix.shape == (4, 2)


def test_lhs_reproducible_random():
    init1 = LatinHypercubeInitializer(2, -1, 1, population_size=4, rng=42)
    init2 = LatinHypercubeInitializer(2, -1, 1, population_size=4, rng=42)
    for _ in range(5):
        v1 = init1.generate_random()
        v2 = init2.generate_random()
        assert_array_equal(v1, v2)


def test_lhs_reproducible_population():
    init1 = LatinHypercubeInitializer(2, -1, 1, population_size=4, rng=42)
    init2 = LatinHypercubeInitializer(2, -1, 1, population_size=4, rng=42)
    for _ in range(5):
        v1 = init1.generate_population()
        v2 = init2.generate_population()
        assert_array_equal(v1.genotype_matrix, v2.genotype_matrix)


# ===================================================================
#  SobolInitializer
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
def test_sobol_generate_random_shape_and_type(genotype_size, low, high, dtype, pop_size, rng):
    init = SobolInitializer(genotype_size, low, high, population_size=pop_size, dtype=dtype, rng=rng)
    vec = init.generate_random()
    assert vec.shape == (genotype_size,)
    assert vec.dtype == np.dtype(dtype)
    assert np.all(vec >= low)
    assert np.all(vec <= high)


def test_sobol_sequence_parameters(rng):
    SobolInitializer(3, [0, 0, 0], [1, 2, 3], rng=rng)
    with pytest.raises(ValueError):
        SobolInitializer(3, [0, 0], [1, 2, 3], rng=rng)
    with pytest.raises(ValueError):
        SobolInitializer(3, [0, 0, 0], [1, 2], rng=rng)


def test_sobol_generate_population(rng):
    init = SobolInitializer(2, -1, 1, population_size=4, rng=rng)
    pop = init.generate_population()
    assert len(pop) == 4
    assert pop.genotype_matrix.shape == (4, 2)


def test_sobol_reproducible_noshuffle():
    init1 = SobolInitializer(2, -1, 1, population_size=4, rng=42)
    init2 = SobolInitializer(2, -1, 1, population_size=4, rng=42)
    for _ in range(5):
        v1 = init1.generate_population()
        v2 = init2.generate_population()
        assert_array_equal(v1.genotype_matrix, v2.genotype_matrix)


def test_sobol_reproducible_scramble():
    init1 = SobolInitializer(2, -1, 1, scramble=True, population_size=4, rng=42)
    init2 = SobolInitializer(2, -1, 1, scramble=True, population_size=4, rng=42)
    for _ in range(5):
        v1 = init1.generate_population()
        v2 = init2.generate_population()
        assert_array_equal(v1.genotype_matrix, v2.genotype_matrix)


# ===================================================================
#  HaltonInitializer
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
def test_halton_generate_random_shape_and_type(genotype_size, low, high, dtype, pop_size, rng):
    init = HaltonInitializer(genotype_size, low, high, population_size=pop_size, dtype=dtype, rng=rng)
    vec = init.generate_random()
    assert vec.shape == (genotype_size,)
    assert vec.dtype == np.dtype(dtype)
    assert np.all(vec >= low)
    assert np.all(vec <= high)


def test_halton_sequence_parameters(rng):
    HaltonInitializer(3, [0, 0, 0], [1, 2, 3], rng=rng)
    with pytest.raises(ValueError):
        HaltonInitializer(3, [0, 0], [1, 2, 3], rng=rng)
    with pytest.raises(ValueError):
        HaltonInitializer(3, [0, 0, 0], [1, 2], rng=rng)


def test_halton_generate_population(rng):
    init = HaltonInitializer(2, -1, 1, population_size=4, rng=rng)
    pop = init.generate_population()
    assert len(pop) == 4
    assert pop.genotype_matrix.shape == (4, 2)


def test_halton_reproducible_noshuffle():
    init1 = HaltonInitializer(2, -1, 1, population_size=4, rng=42)
    init2 = HaltonInitializer(2, -1, 1, population_size=4, rng=42)
    for _ in range(5):
        v1 = init1.generate_population()
        v2 = init2.generate_population()
        assert_array_equal(v1.genotype_matrix, v2.genotype_matrix)


def test_halton_reproducible_scramble():
    init1 = HaltonInitializer(2, -1, 1, scramble=True, population_size=4, rng=42)
    init2 = HaltonInitializer(2, -1, 1, scramble=True, population_size=4, rng=42)
    for _ in range(5):
        v1 = init1.generate_population()
        v2 = init2.generate_population()
        assert_array_equal(v1.genotype_matrix, v2.genotype_matrix)


# ===================================================================
#  SeededInitializer
# ===================================================================
def test_seed_prob_generate_random_returns_valid_shape(rng):
    default_init = UniformInitializer(2, 0, 1, rng=rng)
    seed_init = SeededInitializer(default_init, solutions=np.array([[9, 9]]), insert_prob=0.0, rng=rng)
    vec = seed_init.generate_random()
    assert vec.shape == (2,)
    assert np.all(vec >= 0) and np.all(vec <= 1)


@pytest.mark.parametrize("insert_prob", [0.0, 1.0])
def test_seed_prob_individual_insertion(insert_prob, rng):
    # Use identical seeds for two separate initializers
    rng_default = np.random.default_rng(42)
    rng_seed = np.random.default_rng(42)
    default_init = UniformInitializer(2, 0, 1, rng=rng_default)
    solutions = np.array([[42.0, 42.0]])
    seed_init = SeededInitializer(default_init, solutions=solutions, insert_prob=insert_prob, rng=rng_seed)
    indiv = seed_init.generate_individual()
    if insert_prob == 0.0:
        # Must be a uniform draw, not the seed
        expected_init = UniformInitializer(2, 0, 1, rng=42)
        assert_array_equal(indiv, expected_init.generate_random())
    else:
        assert_array_equal(indiv, [42.0, 42.0])


def test_seed_prob_generate_population_mixed(rng):
    solutions = np.array([[100, 100]])
    default_init = UniformInitializer(2, 0, 1, rng=rng)
    seed_init = SeededInitializer(default_init, solutions=solutions, insert_prob=0.5, rng=42)
    pop = seed_init.generate_population(n_individuals=10)
    assert len(pop) == 10
    assert pop.genotype_matrix.shape == (10, 2)


def test_seed_prob_reproducible_random():
    solutions = np.array([[100, 100]])
    default_init1 = UniformInitializer(2, 0, 1, rng=42)
    init1 = SeededInitializer(default_init1, solutions=solutions, insert_prob=0.5, rng=42)
    default_init2 = UniformInitializer(2, 0, 1, rng=42)
    init2 = SeededInitializer(default_init2, solutions=solutions, insert_prob=0.5, rng=42)
    for _ in range(5):
        v1 = init1.generate_random()
        v2 = init2.generate_random()
        assert_array_equal(v1, v2)


def test_perm_reproducible_population():
    solutions = np.array([[100, 100]])
    default_init1 = UniformInitializer(2, 0, 1, rng=42)
    init1 = SeededInitializer(default_init1, solutions=solutions, insert_prob=0.5, rng=42)
    default_init2 = UniformInitializer(2, 0, 1, rng=42)
    init2 = SeededInitializer(default_init2, solutions=solutions, insert_prob=0.5, rng=42)
    for _ in range(5):
        v1 = init1.generate_population()
        v2 = init2.generate_population()
        assert_array_equal(v1.genotype_matrix, v2.genotype_matrix)


# ===================================================================
#  FixedSeededInitializer
# ===================================================================
def test_seed_determ_inserts_exact_number(rng):
    solutions = np.array([[10, 20], [30, 40]])
    default_init = UniformInitializer(2, 0, 1, rng=rng)
    init = FixedSeededInitializer(default_init, solutions=solutions, n_to_insert=2, rng=rng)
    pop = init.generate_population(n_individuals=5)
    assert_array_equal(pop.genotype_matrix[0], [10, 20])
    assert_array_equal(pop.genotype_matrix[1], [30, 40])
    assert len(pop) == 5
    pop2 = init.generate_population(n_individuals=5)
    assert_array_equal(pop2.genotype_matrix[0], [10, 20])


def test_seed_determ_wraps_around_seed_list(rng):
    solutions = np.array([[1, 1]])
    default_init = UniformInitializer(2, 0, 1, rng=rng)
    init = FixedSeededInitializer(default_init, solutions=solutions, n_to_insert=3, rng=rng)
    pop = init.generate_population(n_individuals=3)
    expected = np.tile([1, 1], (3, 1))
    assert_array_equal(pop.genotype_matrix, expected)


def test_seed_determ_no_insert_inserts_default(rng):
    solutions = np.array([[7, 7]])
    default_init = UniformInitializer(2, 0, 1, rng=rng)
    init = FixedSeededInitializer(default_init, solutions=solutions, n_to_insert=0, rng=rng)
    pop = init.generate_population(n_individuals=4)
    assert not np.any(np.all(pop.genotype_matrix == [7, 7], axis=1))


# ===================================================================
#  DirectInitializer
# ===================================================================
def test_direct_individual_from_array(rng):
    solutions = np.array([[5, 6], [7, 8]])
    default_init = UniformInitializer(2, 0, 1, rng=rng)
    init = DirectInitializer(default_init, solutions, rng=rng)
    indiv = init.generate_individual()
    assert indiv.shape == (2,)
    assert any(np.array_equal(indiv, row) for row in solutions)


def test_direct_individual_from_population(rng):
    solutions = np.array([[1, 2], [3, 4]])
    pop = Population(solutions)
    default_init = UniformInitializer(2, 0, 1, rng=rng)
    init = DirectInitializer(default_init, pop, rng=rng)
    indiv = init.generate_individual()
    assert any(np.array_equal(indiv, row) for row in solutions)


def test_direct_generate_population_from_array(rng):
    solutions = np.array([[10, 20], [30, 40], [50, 60]])
    default_init = UniformInitializer(2, 0, 1, rng=rng)
    init = DirectInitializer(default_init, solutions, rng=rng)
    pop = init.generate_population(n_individuals=5)
    assert len(pop) == 5
    assert_array_equal(pop.genotype_matrix[0], [10, 20])
    assert_array_equal(pop.genotype_matrix[1], [30, 40])
    assert_array_equal(pop.genotype_matrix[2], [50, 60])
    assert_array_equal(pop.genotype_matrix[3], [10, 20])
    assert_array_equal(pop.genotype_matrix[4], [30, 40])


def test_direct_generate_population_from_population_exact(rng):
    solutions = np.array([[1, 1], [2, 2]])
    pop_in = Population(solutions)
    pop_in.fitness = np.array([10.0, 20.0])
    default_init = UniformInitializer(2, 0, 1, rng=rng)
    init = DirectInitializer(default_init, pop_in, rng=rng)
    pop_out = init.generate_population()
    assert_array_equal(pop_out.genotype_matrix, solutions)


# ===================================================================
#  ExtendedInitializer
# ===================================================================
def test_extended_generate_random(rng):
    encoding = DummyParameterExtendingEncoding([("a", 2), ("b", 1)])
    solution_init = UniformInitializer(3, 0, 1, rng=rng)
    param_inits = {
        "a": UniformInitializer(2, 0, 1, rng=np.random.default_rng(100)),
        "b": UniformInitializer(1, 0, 1, rng=np.random.default_rng(101)),
    }
    init = ExtendedInitializer(solution_init, param_inits, encoding, rng=rng)

    # Build a second identical instance for reproducibility check
    rng_expected = np.random.default_rng(42)
    solution_init2 = UniformInitializer(3, 0, 1, rng=np.random.default_rng(42))
    param_inits2 = {
        "a": UniformInitializer(2, 0, 1, rng=np.random.default_rng(100)),
        "b": UniformInitializer(1, 0, 1, rng=np.random.default_rng(101)),
    }
    init2 = ExtendedInitializer(solution_init2, param_inits2, encoding, rng=rng_expected)

    vec = init.generate_random()
    expected = init2.generate_random()
    assert_array_equal(vec, expected)
    assert vec.shape == (6,)


def test_extended_generate_individual_same_structure(rng):
    encoding = DummyParameterExtendingEncoding([("c", 2)])
    solution_init = GaussianInitializer(2, 0, 1, rng=rng)
    param_inits = {"c": GaussianInitializer(2, 5, 0.1, rng=np.random.default_rng(200))}
    init = ExtendedInitializer(solution_init, param_inits, encoding, rng=rng)

    # Second identical instance
    rng_expected = np.random.default_rng(42)
    solution_init2 = GaussianInitializer(2, 0, 1, rng=np.random.default_rng(42))
    param_inits2 = {"c": GaussianInitializer(2, 5, 0.1, rng=np.random.default_rng(200))}
    init2 = ExtendedInitializer(solution_init2, param_inits2, encoding, rng=rng_expected)

    indiv = init.generate_individual()
    expected = init2.generate_individual()
    assert_array_equal(indiv, expected)
    assert indiv.shape == (4,)


# ===================================================================
#  InitializerFromLambda
# ===================================================================
def test_lambda_generate_random(rng):
    def my_gen(rng):
        return rng.uniform(10, 20, size=3)

    init = InitializerFromLambda(my_gen, dimension=3, pop_size=2, rng=rng)
    vec = init.generate_random()
    assert vec.shape == (3,)
    assert np.all(vec >= 10) and np.all(vec <= 20)


def test_lambda_generate_individual_calls_same(rng):
    def my_gen(rng):
        return np.array([1.0, 2.0])

    init = InitializerFromLambda(my_gen, dimension=2, rng=rng)
    assert_array_equal(init.generate_individual(), np.array([1.0, 2.0]))


def test_lambda_generate_population(rng):
    def my_gen(rng):
        return rng.integers(0, 100, size=2)

    init = InitializerFromLambda(my_gen, dimension=2, pop_size=5, rng=rng)
    pop = init.generate_population()
    assert len(pop) == 5
    assert pop.genotype_matrix.shape == (5, 2)
    assert np.issubdtype(pop.genotype_matrix.dtype, np.integer)
    assert np.all(pop.genotype_matrix >= 0)
    assert np.all(pop.genotype_matrix < 100)


# ===================================================================
#  CompositeInitializer (probabilistic / weighted)
# ===================================================================
def test_composite_generate_individual_distribution(rng):
    """Check that over many calls the proportion of selected initializers
    approximates the given weights."""
    init_a = UniformInitializer(2, 0, 1, rng=rng)
    init_b = GaussianInitializer(2, 5, 1, rng=rng)
    weights = [0.7, 0.3]
    comp = CompositeInitializer(2, [init_a, init_b], weights=weights, rng=rng)

    n_samples = 10000
    counts = {0: 0, 1: 0}
    for _ in range(n_samples):
        indiv = comp.generate_individual()
        # Check shape only, we don't know the exact values; we need to know which init produced it.
        # We can use a trick: the uniform init gives values in [0,1], Gaussian around 5.
        # So classify based on first element.
        if indiv[0] < 2:
            counts[0] += 1
        else:
            counts[1] += 1
    expected_0 = n_samples * weights[0]
    # Allow some tolerance (e.g., 5% relative error)
    assert abs(counts[0] - expected_0) / n_samples < 0.05


def test_composite_generate_population_individual_wise(rng):
    """Verify that each individual in a population comes entirely from one
    initializer (not mixed gene‑wise)."""
    init_a = UniformInitializer(2, 0, 1, rng=rng)
    init_b = GaussianInitializer(2, 100, 0.1, rng=rng)  # very different values
    weights = [0.5, 0.5]
    comp = CompositeInitializer(2, [init_a, init_b], weights=weights, rng=rng)
    pop = comp.generate_population(n_individuals=100)

    # For each row, all values should either be from init_a (low, in [0,1]) or init_b (near 100)
    for row in pop.genotype_matrix:
        if row[0] < 50:  # from uniform
            assert np.all(row >= 0) and np.all(row <= 1)
        else:  # from gaussian
            assert np.all(row > 99) and np.all(row < 101)


def test_composite_reproducible_random():
    init1_a = UniformInitializer(2, 0, 1, rng=42)
    init1_b = GaussianInitializer(2, 5, 1, rng=42)
    comp1 = CompositeInitializer(2, [init1_a, init1_b], weights=[0.5, 0.5], rng=42)

    init2_a = UniformInitializer(2, 0, 1, rng=42)
    init2_b = GaussianInitializer(2, 5, 1, rng=42)
    comp2 = CompositeInitializer(2, [init2_a, init2_b], weights=[0.5, 0.5], rng=42)
    for _ in range(10):
        v1 = comp1.generate_random()
        v2 = comp2.generate_random()
        assert_array_equal(v1, v2)


def test_composite_reproducible_population(dummy_objfunc):
    init1_a = UniformInitializer(2, 0, 1, rng=42)
    init1_b = GaussianInitializer(2, 5, 1, rng=42)
    comp1 = CompositeInitializer(2, [init1_a, init1_b], weights=[0.5, 0.5], rng=42)

    init2_a = UniformInitializer(2, 0, 1, rng=42)
    init2_b = GaussianInitializer(2, 5, 1, rng=42)
    comp2 = CompositeInitializer(2, [init2_a, init2_b], weights=[0.5, 0.5], rng=42)
    for _ in range(5):
        pop1 = comp1.generate_population(n_individuals=20)
        pop2 = comp2.generate_population(n_individuals=20)
        assert_array_equal(pop1.genotype_matrix, pop2.genotype_matrix)


# ===================================================================
#  FixedCompositeInitializer (deterministic exact counts)
# ===================================================================
def test_fixed_composite_individual_cycle(rng):
    init_a = UniformInitializer(1, 0, 1, rng=rng)
    init_b = GaussianInitializer(1, 10, 1, rng=rng)
    init_c = UniformInitializer(1, 100, 101, rng=rng)
    amounts = [2, 1, 3]  # total 6
    fixed = FixedCompositeInitializer(1, [init_a, init_b, init_c], amounts=amounts, rng=rng)

    expected_sequence = [0, 0, 1, 2, 2, 2]  # indices from amounts
    for i in range(6):
        indiv = fixed.generate_individual()
        # Classify based on value range
        if indiv[0] < 1:
            idx = 0
        elif indiv[0] < 20:
            idx = 1
        else:
            idx = 2
        assert idx == expected_sequence[i]
    # After one full cycle, the pattern repeats
    indiv = fixed.generate_individual()
    if indiv[0] < 1:
        idx = 0
    elif indiv[0] < 20:
        idx = 1
    else:
        idx = 2
    assert idx == expected_sequence[0]


def test_fixed_composite_population_cycle(rng):
    init_a = UniformInitializer(1, 0, 1, rng=rng)
    init_b = UniformInitializer(1, 100, 101, rng=rng)
    amounts = [3, 2]  # total 5
    fixed = FixedCompositeInitializer(1, [init_a, init_b], amounts=amounts, rng=rng)

    # Generate a population of 8 individuals (more than total)
    pop = fixed.generate_population(n_individuals=8)
    # Expected pattern: 3 from init_a, then 2 from init_b, then repeat: 3 from init_a, 2 from init_b
    # But the first 3 from init_a will be values < 1, next 2 from init_b will be > 99, then next 3 from init_a again, etc.
    gen_matrix = pop.genotype_matrix.flatten()
    expected_pattern = ([0] * 3 + [1] * 2) * 2  # up to 8: 0,0,0,1,1,0,0,0  (last group truncated to 3 from init_a)
    # Check first 8
    for i, expected in enumerate(expected_pattern[:8]):
        if expected == 0:
            assert 0 <= gen_matrix[i] <= 1
        else:
            assert 100 <= gen_matrix[i] <= 101


def test_fixed_composite_amounts_sum_to_population():
    """If amounts are provided, they should be used exactly; population_size
    is automatically the sum of amounts."""
    init_a = UniformInitializer(1, 0, 1, rng=42)
    init_b = UniformInitializer(1, 100, 101, rng=42)  # was Gaussian
    amounts = [5, 5]
    fixed = FixedCompositeInitializer(1, [init_a, init_b], amounts=amounts, rng=42)
    pop = fixed.generate_population()
    assert len(pop) == 10
    first5 = pop.genotype_matrix[:5]
    last5 = pop.genotype_matrix[5:]
    assert np.all(first5 >= 0) and np.all(first5 <= 1)
    assert np.all(last5 >= 100) and np.all(last5 <= 101)  # now always true


def test_fixed_composite_reproducible():
    init1_a = UniformInitializer(1, 0, 1, rng=42)
    init1_b = GaussianInitializer(1, 10, 1, rng=42)
    fixed1 = FixedCompositeInitializer(1, [init1_a, init1_b], amounts=[2, 3], rng=42)

    init2_a = UniformInitializer(1, 0, 1, rng=42)
    init2_b = GaussianInitializer(1, 10, 1, rng=42)
    fixed2 = FixedCompositeInitializer(1, [init2_a, init2_b], amounts=[2, 3], rng=42)
    for _ in range(10):
        v1 = fixed1.generate_individual()
        v2 = fixed2.generate_individual()
        assert_array_equal(v1, v2)


def test_fixed_composite_population_reproducible():
    init1_a = UniformInitializer(1, 0, 1, rng=42)
    init1_b = GaussianInitializer(1, 10, 1, rng=42)
    fixed1 = FixedCompositeInitializer(1, [init1_a, init1_b], amounts=[2, 3], rng=42)

    init2_a = UniformInitializer(1, 0, 1, rng=42)
    init2_b = GaussianInitializer(1, 10, 1, rng=42)
    fixed2 = FixedCompositeInitializer(1, [init2_a, init2_b], amounts=[2, 3], rng=42)
    pop1 = fixed1.generate_population(n_individuals=12)
    pop2 = fixed2.generate_population(n_individuals=12)
    assert_array_equal(pop1.genotype_matrix, pop2.genotype_matrix)
