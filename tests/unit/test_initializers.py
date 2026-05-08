"""
Unit tests for Initializer hierarchy.

Contracts verified:
- generate_individual() returns a vector of the correct dimension.
- generate_population() returns a Population of the correct size and dimension.
- UniformInitializer output lies within declared bounds.
- GaussianInitializer returns finite floats.
- PermInitializer returns permutations of [0, n-1].
- SeedDetermInitializer always returns the same individual.
"""

import numpy as np
import pytest

from metaheuristic_designer.initializers import (
    UniformInitializer,
    GaussianInitializer,
    PermInitializer,
)
from metaheuristic_designer.initializers.seed_initializer import SeedDetermInitializer, SeedProbInitializer
from metaheuristic_designer.initializers.direct_initializer import DirectInitializer
from metaheuristic_designer.benchmarks.benchmark_funcs import MaxOnes, Sphere


# ---------------------------------------------------------------------------
# UniformInitializer
# ---------------------------------------------------------------------------

def test_uniform_initializer_individual_dimension():
    init = UniformInitializer(dimension=6, lower_bound=np.zeros(6), upper_bound=np.ones(6))
    ind = init.generate_individual()
    assert np.asarray(ind).shape == (6,)


def test_uniform_initializer_individual_within_bounds():
    lb = np.full(5, -3.0)
    ub = np.full(5, 3.0)
    init = UniformInitializer(dimension=5, lower_bound=lb, upper_bound=ub, random_state=0)
    for _ in range(20):
        ind = np.asarray(init.generate_individual())
        assert np.all(ind >= lb)
        assert np.all(ind <= ub)


def test_uniform_initializer_population_size():
    init = UniformInitializer(
        dimension=4, lower_bound=np.zeros(4), upper_bound=np.ones(4), pop_size=8
    )
    objfunc = MaxOnes(dimension=4)
    pop = init.generate_population(objfunc)
    assert pop.pop_size == 8


def test_uniform_initializer_population_within_bounds():
    lb = np.full(4, -1.0)
    ub = np.full(4, 1.0)
    init = UniformInitializer(
        dimension=4, lower_bound=lb, upper_bound=ub, pop_size=20, random_state=7
    )
    objfunc = Sphere(dimension=4, mode="min")
    pop = init.generate_population(objfunc)
    assert np.all(pop.genotype_matrix >= lb)
    assert np.all(pop.genotype_matrix <= ub)


# ---------------------------------------------------------------------------
# GaussianInitializer
# ---------------------------------------------------------------------------

def test_gaussian_initializer_individual_shape():
    init = GaussianInitializer(dimension=5, g_mean=0.0, g_std=1.0)
    ind = np.asarray(init.generate_individual())
    assert ind.shape == (5,)


def test_gaussian_initializer_individual_is_finite():
    init = GaussianInitializer(dimension=10, g_mean=0.0, g_std=1.0, random_state=42)
    for _ in range(10):
        ind = np.asarray(init.generate_individual())
        assert np.all(np.isfinite(ind))


def test_gaussian_initializer_population_size():
    init = GaussianInitializer(dimension=3, g_mean=0.0, g_std=1.0, pop_size=12)
    objfunc = Sphere(dimension=3, mode="min")
    pop = init.generate_population(objfunc)
    assert pop.pop_size == 12


# ---------------------------------------------------------------------------
# PermInitializer
# ---------------------------------------------------------------------------

def test_perm_initializer_individual_is_permutation():
    n = 8
    init = PermInitializer(dimension=n)
    ind = np.asarray(init.generate_individual())
    assert ind.shape == (n,)
    assert sorted(ind.tolist()) == list(range(n))


def test_perm_initializer_population_all_permutations():
    n = 6
    init = PermInitializer(dimension=n, population_size=10, random_state=3)
    objfunc = MaxOnes(dimension=n)
    pop = init.generate_population(objfunc)
    for row in pop.genotype_matrix:
        assert sorted(row.astype(int).tolist()) == list(range(n))


# ---------------------------------------------------------------------------
# SeedDetermInitializer – always inserts predefined solutions deterministically
# ---------------------------------------------------------------------------

def test_seed_determ_initializer_always_same_individual():
    """SeedDetermInitializer must always return the seeded solutions."""
    base_init = UniformInitializer(dimension=4, lower_bound=np.zeros(4), upper_bound=np.ones(4), pop_size=5)
    seed_vecs = np.array([[1.0, 2.0, 3.0, 4.0], [0.5, 0.5, 0.5, 0.5]])
    init = SeedDetermInitializer(default_init=base_init, solutions=seed_vecs)
    # All individuals must come from the seed vectors
    ind1 = np.asarray(init.generate_individual())
    ind2 = np.asarray(init.generate_individual())
    # Each returned individual must be one of the seed vectors
    all_seeds = set(map(tuple, seed_vecs.tolist()))
    assert tuple(ind1.tolist()) in all_seeds
    assert tuple(ind2.tolist()) in all_seeds


# ---------------------------------------------------------------------------
# GaussianInitializer with vector g_mean and g_std
# ---------------------------------------------------------------------------

def test_gaussian_initializer_vector_mean_std():
    mean = [0.0, 1.0, 2.0, 3.0]
    std = [0.1, 0.2, 0.3, 0.4]
    init = GaussianInitializer(dimension=4, g_mean=mean, g_std=std, random_state=0)
    ind = np.asarray(init.generate_individual())
    assert ind.shape == (4,)
    assert np.all(np.isfinite(ind))


def test_gaussian_initializer_wrong_mean_length_raises():
    with pytest.raises(ValueError, match="g_mean"):
        GaussianInitializer(dimension=4, g_mean=[1.0, 2.0], g_std=1.0)


def test_gaussian_initializer_wrong_std_length_raises():
    with pytest.raises(ValueError, match="g_std"):
        GaussianInitializer(dimension=4, g_mean=0.0, g_std=[0.1, 0.2])


# ---------------------------------------------------------------------------
# DirectInitializer
# ---------------------------------------------------------------------------

def test_direct_initializer_from_ndarray():
    base_init = UniformInitializer(dimension=4, lower_bound=-1.0, upper_bound=1.0, pop_size=5)
    solutions = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    init = DirectInitializer(default_init=base_init, solutions=solutions)
    ind = init.generate_individual()
    assert np.asarray(ind).shape == (4,)


def test_direct_initializer_generate_population_from_ndarray():
    base_init = UniformInitializer(dimension=4, lower_bound=-1.0, upper_bound=1.0, pop_size=5)
    solutions = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    init = DirectInitializer(default_init=base_init, solutions=solutions, random_state=0)
    objfunc = Sphere(dimension=4, mode="min")
    pop = init.generate_population(objfunc, n_individuals=2)
    assert pop.pop_size == 2


def test_direct_initializer_from_population():
    from metaheuristic_designer.population import Population
    base_init = UniformInitializer(dimension=4, lower_bound=-1.0, upper_bound=1.0, pop_size=3)
    objfunc = Sphere(dimension=4, mode="min")
    geno = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [-1.0, -2.0, -3.0, -4.0]])
    solutions_pop = Population(objfunc, geno)
    init = DirectInitializer(default_init=base_init, solutions=solutions_pop, random_state=1)
    pop = init.generate_population(objfunc, n_individuals=3)
    assert pop.pop_size == 3


def test_direct_initializer_generate_random_delegates_to_base():
    base_init = UniformInitializer(dimension=4, lower_bound=-1.0, upper_bound=1.0, pop_size=3,
                                    random_state=0)
    solutions = np.ones((2, 4))
    init = DirectInitializer(default_init=base_init, solutions=solutions)
    rnd = init.generate_random()
    assert np.asarray(rnd).shape == (4,)


def test_direct_initializer_population_different_size():
    """generate_population with n_individuals != solutions.pop_size uses cycling."""
    base_init = UniformInitializer(dimension=3, lower_bound=-1.0, upper_bound=1.0, pop_size=5)
    solutions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    init = DirectInitializer(default_init=base_init, solutions=solutions, random_state=2)
    objfunc = Sphere(dimension=3, mode="min")
    pop = init.generate_population(objfunc, n_individuals=5)
    assert pop.pop_size == 5


def test_direct_initializer_wrong_type_raises():
    """DirectInitializer raises TypeError if solutions is not Population or ndarray."""
    base_init = UniformInitializer(dimension=4, lower_bound=-1.0, upper_bound=1.0, pop_size=3)
    # List-of-lists not supported in generate_individual
    solutions = np.ones((2, 4))  # valid for construction
    init = DirectInitializer(default_init=base_init, solutions=solutions, random_state=3)
    # Replace solutions with invalid type
    init.solutions = [[1.0, 2.0, 3.0, 4.0]]  # list-of-lists → invalid
    with pytest.raises(TypeError):
        init.generate_individual()


# ---------------------------------------------------------------------------
# SeedProbInitializer
# ---------------------------------------------------------------------------

def test_seed_prob_initializer_with_high_probability_returns_seed():
    """With insert_prob=1.0, always returns a seeded solution."""
    base_init = UniformInitializer(dimension=4, lower_bound=-1.0, upper_bound=1.0, pop_size=5,
                                    random_state=0)
    seed_vecs = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    init = SeedProbInitializer(default_init=base_init, solutions=seed_vecs,
                                insert_prob=1.0, random_state=0)
    for _ in range(5):
        ind = np.asarray(init.generate_individual())
        all_seeds = [tuple(s) for s in seed_vecs.tolist()]
        assert tuple(ind.tolist()) in all_seeds


def test_seed_prob_initializer_with_zero_probability_returns_random():
    """With insert_prob=0.0, always returns a random solution (not the seeded one)."""
    base_init = UniformInitializer(dimension=4, lower_bound=-1.0, upper_bound=1.0, pop_size=5,
                                    random_state=42)
    seed_vecs = np.array([[1000.0, 2000.0, 3000.0, 4000.0]])
    init = SeedProbInitializer(default_init=base_init, solutions=seed_vecs,
                                insert_prob=0.0, random_state=42)
    ind = np.asarray(init.generate_individual())
    assert not np.array_equal(ind, seed_vecs[0])


def test_seed_prob_initializer_population_size():
    base_init = UniformInitializer(dimension=4, lower_bound=-1.0, upper_bound=1.0, pop_size=8,
                                    random_state=7)
    seed_vecs = np.array([[0.5, 0.5, 0.5, 0.5]])
    init = SeedProbInitializer(default_init=base_init, solutions=seed_vecs,
                                insert_prob=0.5, random_state=7)
    objfunc = Sphere(dimension=4, mode="min")
    pop = init.generate_population(objfunc)
    assert pop.pop_size == 8


@pytest.mark.xfail(
    reason="BUG: SeedProbInitializer.generate_individual() uses random_state.choice(self.solutions, axis=0) "
           "which fails when solutions is a Population object (ValueError: a must be a sequence or an integer). "
           "Unlike SeedDetermInitializer which correctly accesses .genotype_matrix, SeedProbInitializer does not "
           "handle Population objects. See ERRORES.md."
)
def test_seed_prob_initializer_from_population():
    """SeedProbInitializer also accepts a Population as solutions."""
    from metaheuristic_designer.population import Population
    base_init = UniformInitializer(dimension=4, lower_bound=-1.0, upper_bound=1.0, pop_size=5,
                                    random_state=3)
    objfunc = Sphere(dimension=4, mode="min")
    geno = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    solutions_pop = Population(objfunc, geno)
    init = SeedProbInitializer(default_init=base_init, solutions=solutions_pop,
                                insert_prob=0.8, random_state=3)
    ind = np.asarray(init.generate_individual())
    assert ind.shape == (4,)
