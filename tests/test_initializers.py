import pytest
import numpy as np

# Adjust imports to your actual project
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


# -------------------------------------------------------------------
#  Helpers that return expected arrays for a known seed (42)
# -------------------------------------------------------------------

def _expected_exponential(beta, size, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    return rng.exponential(beta, size=size)

def _expected_normal(mean, std, size, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    return rng.normal(mean, std, size=size)

def _expected_uniform(low, high, size, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    return rng.uniform(low, high, size=size)

def _expected_permutation(n, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    return rng.permutation(n)


# ===================================================================
#  ExponentialInitializer
# ===================================================================

@pytest.mark.parametrize("genotype_size, beta, dtype, pop_size", [
    (3, 1.0, float, 1),
    (5, 2.0, float, 4),
    (4, 0.5, int, 1),
    (3, 1.5, int, 2),
    (1, 10.0, float, 1),
])
def test_exponential_generate_random_shape_and_type(genotype_size, beta, dtype, pop_size, rng):
    init = ExponentialInitializer(genotype_size, beta, pop_size=pop_size, dtype=dtype, random_state=rng)
    vec = init.generate_random()
    assert vec.shape == (genotype_size,)
    assert vec.dtype == np.dtype(dtype)
    assert np.all(vec >= 0)   # exponential is always non‑negative


@pytest.mark.parametrize("genotype_size, beta, dtype, pop_size", [
    (3, 1.0, float, 1),
    (2, 2.5, int, 1),
])
def test_exponential_generate_random_deterministic(genotype_size, beta, dtype, pop_size, rng):
    init = ExponentialInitializer(genotype_size, beta, pop_size=pop_size, dtype=dtype, random_state=rng)
    expected = _expected_exponential(beta, genotype_size)
    if dtype is int:
        expected = np.round(expected).astype(dtype)
    else:
        expected = expected.astype(dtype)
    np.testing.assert_array_almost_equal(init.generate_random(), expected)


def test_exponential_generate_individual_same_as_random(rng):
    init = ExponentialInitializer(3, 1.0, random_state=rng)
    np.testing.assert_array_equal(init.generate_individual(), init.generate_random())


@pytest.mark.parametrize("pop_size", [1, 4])
def test_exponential_generate_population(pop_size, rng, objfunc, encoding):
    init = ExponentialInitializer(2, 1.0, pop_size=pop_size, encoding=encoding, random_state=rng)
    pop = init.generate_population(objfunc)
    assert len(pop) == pop_size
    assert pop.genotype_matrix.shape == (pop_size, 2)
    assert pop.objfunc is objfunc
    assert pop.encoding is encoding


# ===================================================================
#  GaussianInitializer
# ===================================================================

@pytest.mark.parametrize("genotype_size, g_mean, g_std, dtype, pop_size", [
    (3, 0.0, 1.0, float, 1),
    (4, 5.0, 0.5, float, 2),
    (2, 10.0, 0.1, int, 1),
    (1, 0.0, 2.0, float, 1),
])
def test_gaussian_generate_random_shape_and_type(genotype_size, g_mean, g_std, dtype, pop_size, rng):
    init = GaussianInitializer(genotype_size, g_mean, g_std, pop_size=pop_size, dtype=dtype, random_state=rng)
    vec = init.generate_random()
    assert vec.shape == (genotype_size,)
    assert vec.dtype == np.dtype(dtype)


@pytest.mark.parametrize("genotype_size, g_mean, g_std, dtype", [
    (2, 0.0, 1.0, float),
    (3, 2.0, 0.5, int),
])
def test_gaussian_generate_random_deterministic(genotype_size, g_mean, g_std, dtype, rng):
    init = GaussianInitializer(genotype_size, g_mean, g_std, pop_size=1, dtype=dtype, random_state=rng)
    expected = _expected_normal(g_mean, g_std, genotype_size)
    if dtype is int:
        expected = np.round(expected).astype(dtype)
    else:
        expected = expected.astype(dtype)
    np.testing.assert_array_almost_equal(init.generate_random(), expected)


def test_gaussian_generate_individual_same_as_random(rng):
    init = GaussianInitializer(3, 0, 1, random_state=rng)
    np.testing.assert_array_equal(init.generate_individual(), init.generate_random())


def test_gaussian_sequence_parameters(rng):
    # Array mean/std must match genotype_size
    GaussianInitializer(3, [1, 2, 3], [0.1, 0.2, 0.3], random_state=rng)
    with pytest.raises(ValueError):
        GaussianInitializer(3, [1, 2], [0.1, 0.2, 0.3], random_state=rng)
    with pytest.raises(ValueError):
        GaussianInitializer(3, [1, 2, 3], [0.1, 0.2], random_state=rng)


def test_gaussian_generate_population(rng, objfunc):
    init = GaussianInitializer(2, 1.0, 0.2, pop_size=5, random_state=rng)
    pop = init.generate_population(objfunc)
    assert len(pop) == 5
    assert pop.genotype_matrix.shape == (5, 2)


# ===================================================================
#  UniformInitializer
# ===================================================================

@pytest.mark.parametrize("genotype_size, low, high, dtype, pop_size", [
    (3, -1.0, 1.0, float, 1),
    (4, 0.0, 10.0, float, 3),
    (2, 2.0, 5.0, int, 1),
    (1, 0.0, 100.0, float, 1),
])
def test_uniform_generate_random_shape_and_type(genotype_size, low, high, dtype, pop_size, rng):
    init = UniformInitializer(genotype_size, low, high, pop_size=pop_size, dtype=dtype, random_state=rng)
    vec = init.generate_random()
    assert vec.shape == (genotype_size,)
    assert vec.dtype == np.dtype(dtype)
    # Check bounds (float will be within, int after rounding may hit edges)
    assert np.all(vec >= low)
    assert np.all(vec <= high)


@pytest.mark.parametrize("genotype_size, low, high, dtype", [
    (2, 0.0, 1.0, float),
    (3, 5.0, 10.0, int),
])
def test_uniform_generate_random_deterministic(genotype_size, low, high, dtype, rng):
    init = UniformInitializer(genotype_size, low, high, pop_size=1, dtype=dtype, random_state=rng)
    expected = _expected_uniform(low, high, genotype_size)
    if dtype is int:
        expected = np.round(expected).astype(dtype)
    else:
        expected = expected.astype(dtype)
    np.testing.assert_array_almost_equal(init.generate_random(), expected)


def test_uniform_generate_individual_same_as_random(rng):
    init = UniformInitializer(3, 0, 1, random_state=rng)
    np.testing.assert_array_equal(init.generate_individual(), init.generate_random())


def test_uniform_sequence_parameters(rng):
    UniformInitializer(3, [0, 0, 0], [1, 2, 3], random_state=rng)
    with pytest.raises(ValueError):
        UniformInitializer(3, [0, 0], [1, 2, 3], random_state=rng)
    with pytest.raises(ValueError):
        UniformInitializer(3, [0, 0, 0], [1, 2], random_state=rng)


def test_uniform_generate_population(rng, objfunc):
    init = UniformInitializer(2, -1, 1, pop_size=4, random_state=rng)
    pop = init.generate_population(objfunc)
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
    # Must be a permutation of 0..n-1
    np.testing.assert_array_equal(np.sort(perm), np.arange(n))


def test_perm_generate_random_deterministic(rng):
    init = PermInitializer(5, random_state=rng)
    expected = _expected_permutation(5)
    np.testing.assert_array_equal(init.generate_random(), expected)


def test_perm_generate_individual_same_as_random(rng):
    init = PermInitializer(4, random_state=rng)
    np.testing.assert_array_equal(init.generate_individual(), init.generate_random())


def test_perm_generate_population(rng, objfunc):
    init = PermInitializer(3, pop_size=5, random_state=rng)
    pop = init.generate_population(objfunc)
    assert len(pop) == 5
    assert pop.genotype_matrix.shape == (5, 3)
    # Each row must be a permutation
    for row in pop.genotype_matrix:
        np.testing.assert_array_equal(np.sort(row), np.arange(3))


# ===================================================================
#  SeedProbInitializer
# ===================================================================

def test_seed_prob_generate_random_uses_default(rng):
    default_init = UniformInitializer(2, 0, 1, random_state=rng)
    seed_init = SeedProbInitializer(default_init, solutions=np.array([[9,9]]), insert_prob=0.0, random_state=rng)
    # When prob is 0, generate_random still calls default
    np.testing.assert_array_equal(seed_init.generate_random(), default_init.generate_random())


@pytest.mark.parametrize("insert_prob", [0.0, 1.0])
def test_seed_prob_individual_insertion(insert_prob, rng):
    default_init = UniformInitializer(2, 0, 1, random_state=rng)
    solutions = np.array([[42.0, 42.0]])
    seed_init = SeedProbInitializer(default_init, solutions=solutions, insert_prob=insert_prob, random_state=rng)
    indiv = seed_init.generate_individual()
    if insert_prob == 0.0:
        # Should be uniform, not the seed
        assert not np.allclose(indiv, [42.0, 42.0])
    else:
        np.testing.assert_array_equal(indiv, [42.0, 42.0])


def test_seed_prob_generate_population_mixed(rng, objfunc):
    # Use a relatively high prob so we almost certainly get a mix
    solutions = np.array([[100, 100]])
    default_init = UniformInitializer(2, 0, 1, random_state=rng)
    seed_init = SeedProbInitializer(default_init, solutions=solutions, insert_prob=0.5, random_state=rng)
    pop = seed_init.generate_population(objfunc, n_individuals=10)
    assert len(pop) == 10
    assert pop.genotype_matrix.shape == (10, 2)


# ===================================================================
#  SeedDetermInitializer
# ===================================================================

def test_seed_determ_inserts_exact_number(rng, objfunc):
    solutions = np.array([[10, 20], [30, 40]])
    default_init = UniformInitializer(2, 0, 1, random_state=rng)
    init = SeedDetermInitializer(default_init, solutions=solutions, n_to_insert=2, random_state=rng)
    pop = init.generate_population(objfunc, n_individuals=5)
    # First 2 individuals must be the solutions in order
    np.testing.assert_array_equal(pop.genotype_matrix[0], [10, 20])
    np.testing.assert_array_equal(pop.genotype_matrix[1], [30, 40])
    # The rest (3) are random (not from solutions)
    assert len(pop) == 5
    # generate_population resets the counter; repeat call should behave the same
    pop2 = init.generate_population(objfunc, n_individuals=5)
    np.testing.assert_array_equal(pop2.genotype_matrix[0], [10, 20])


def test_seed_determ_wraps_around_seed_list(rng, objfunc):
    solutions = np.array([[1, 1]])
    default_init = UniformInitializer(2, 0, 1, random_state=rng)
    init = SeedDetermInitializer(default_init, solutions=solutions, n_to_insert=3, random_state=rng)
    pop = init.generate_population(objfunc, n_individuals=3)
    # Only one solution, so it repeats
    expected = np.tile([1, 1], (3, 1))
    np.testing.assert_array_equal(pop.genotype_matrix, expected)


def test_seed_determ_no_insert_inserts_default(rng, objfunc):
    solutions = np.array([[7, 7]])
    default_init = UniformInitializer(2, 0, 1, random_state=rng)
    init = SeedDetermInitializer(default_init, solutions=solutions, n_to_insert=0, random_state=rng)
    pop = init.generate_population(objfunc, n_individuals=4)
    # None should be the seed
    assert not np.any(np.all(pop.genotype_matrix == [7, 7], axis=1))


# ===================================================================
#  DirectInitializer
# ===================================================================

def test_direct_individual_from_array(rng):
    solutions = np.array([[5, 6], [7, 8]])
    default_init = UniformInitializer(2, 0, 1, random_state=rng)
    init = DirectInitializer(default_init, solutions, random_state=rng)
    indiv = init.generate_individual()
    # It should be one of the rows (random choice with fixed seed)
    assert indiv.shape == (2,)
    assert any(np.array_equal(indiv, row) for row in solutions)


def test_direct_individual_from_population(rng):
    solutions = np.array([[1, 2], [3, 4]])
    pop = Population(DummyObjectiveFunction("dummy"), solutions)
    default_init = UniformInitializer(2, 0, 1, random_state=rng)
    init = DirectInitializer(default_init, pop, random_state=rng)
    indiv = init.generate_individual()
    assert any(np.array_equal(indiv, row) for row in solutions)


def test_direct_generate_population_from_array(rng, objfunc):
    solutions = np.array([[10, 20], [30, 40], [50, 60]])
    default_init = UniformInitializer(2, 0, 1, random_state=rng)
    init = DirectInitializer(default_init, solutions, random_state=rng)
    pop = init.generate_population(objfunc, n_individuals=5)
    # Population size should be 5, but solutions only 3 → they cycle
    assert len(pop) == 5
    # First three should be the solutions in order (cycling)
    np.testing.assert_array_equal(pop.genotype_matrix[0], [10, 20])
    np.testing.assert_array_equal(pop.genotype_matrix[1], [30, 40])
    np.testing.assert_array_equal(pop.genotype_matrix[2], [50, 60])
    np.testing.assert_array_equal(pop.genotype_matrix[3], [10, 20])  # wraps
    np.testing.assert_array_equal(pop.genotype_matrix[4], [30, 40])


def test_direct_generate_population_from_population_exact(rng, objfunc):
    solutions = np.array([[1, 1], [2, 2]])
    pop_in = Population(objfunc, solutions)
    pop_in.fitness = np.array([10.0, 20.0])  # some state
    default_init = UniformInitializer(2, 0, 1, random_state=rng)
    init = DirectInitializer(default_init, pop_in, random_state=rng)
    pop_out = init.generate_population(objfunc, n_individuals=2)
    assert len(pop_out) == 2
    np.testing.assert_array_equal(pop_out.genotype_matrix, solutions)
    # Should have copied the fitness from the input population
    np.testing.assert_array_equal(pop_out.fitness, pop_in.fitness)


# ===================================================================
#  ExtendedInitializer
# ===================================================================

# To test ExtendedInitializer we need a mock ParameterExtendingEncoding.
# We'll create a minimal one without importing the real one to avoid side effects.

class MockParameterExtendingEncoding:
    def __init__(self, param_sizes):
        self.param_sizes = param_sizes  # list of (name, size)


def test_extended_generate_random(rng):
    encoding = MockParameterExtendingEncoding([("a", 2), ("b", 1)])
    solution_init = UniformInitializer(3, 0, 1, random_state=rng)
    param_inits = {
        "a": UniformInitializer(2, 0, 1, random_state=rng),
        "b": UniformInitializer(1, 0, 1, random_state=rng),
    }
    init = ExtendedInitializer(solution_init, param_inits, encoding, random_state=rng)
    vec = init.generate_random()
    assert vec.shape == (3 + 2 + 1,)
    # Just check it was built by stacking three parts (we can test exact with seeds)
    # Using known seed we can recompose expected value
    expected_sol = solution_init.generate_random()
    expected_a = param_inits["a"].generate_random()
    expected_b = param_inits["b"].generate_random()
    expected = np.hstack([expected_sol, expected_a, expected_b])
    np.testing.assert_array_almost_equal(vec, expected)


def test_extended_generate_individual_same_structure(rng):
    encoding = MockParameterExtendingEncoding([("c", 2)])
    solution_init = GaussianInitializer(2, 0, 1, random_state=rng)
    param_inits = {"c": GaussianInitializer(2, 5, 0.1, random_state=rng)}
    init = ExtendedInitializer(solution_init, param_inits, encoding, random_state=rng)
    indiv = init.generate_individual()
    assert indiv.shape == (2 + 2,)
    expected_sol = solution_init.generate_individual()
    expected_c = param_inits["c"].generate_individual()
    np.testing.assert_array_almost_equal(indiv, np.hstack([expected_sol, expected_c]))


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
    # A deterministic lambda
    def my_gen(rs):
        return np.array([1.0, 2.0])
    init = InitializerFromLambda(my_gen, random_state=rng)
    np.testing.assert_array_equal(init.generate_individual(), np.array([1.0, 2.0]))


def test_lambda_generate_population(rng, objfunc):
    def my_gen(rs):
        return rs.integers(0, 100, size=2)
    init = InitializerFromLambda(my_gen, pop_size=5, random_state=rng)
    pop = init.generate_population(objfunc)
    assert len(pop) == 5
    assert pop.genotype_matrix.shape == (5, 2)
    # all values should be integers between 0 and 99
    assert np.issubdtype(pop.genotype_matrix.dtype, np.integer)
    assert np.all(pop.genotype_matrix >= 0)
    assert np.all(pop.genotype_matrix < 100)