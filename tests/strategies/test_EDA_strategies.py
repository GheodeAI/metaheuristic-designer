import numpy as np
from numpy.testing import assert_array_equal

from conftest import rng, dummy_objfunc, dummy_initializer

from metaheuristic_designer.strategies.EDA.cross_entropy_method import CrossEntropyMethod
from metaheuristic_designer.strategies.EDA.PBIL import (
    BernoulliPBIL,
    BinomialPBIL,
    GaussianPBIL,
)
from metaheuristic_designer.strategies.EDA.UMDA import (
    BernoulliUMDA,
    BinomialUMDA,
    GaussianUMDA,
)
from metaheuristic_designer.population import Population


# -------------------------------------------------------------------
#  CrossEntropyMethod
# -------------------------------------------------------------------
def test_cross_entropy_method_creation(rng, dummy_initializer):
    algo = CrossEntropyMethod(initializer=dummy_initializer, elite_amount=5, random_state=rng)
    assert algo.name == "CrossEntropyMethod"
    assert algo.operator is not None
    assert algo.parent_sel is not None


# -------------------------------------------------------------------
#  BernoulliPBIL
# -------------------------------------------------------------------
def test_bernoulli_pbil_perturb_updates_p(rng, dummy_objfunc, dummy_initializer):
    pop = Population(dummy_objfunc, np.ones((3, 2)))
    pop.fitness = np.zeros(3)

    # Compute expected mean BEFORE perturb modifies pop
    expected_p = pop.genotype_matrix.mean(axis=0)  # [1., 1.]

    algo = BernoulliPBIL(
        initializer=dummy_initializer,
        p=0.5,
        noise=0,
        lr=1.0,
        random_state=rng,
    )

    algo.estimate_parameters(pop)

    actual_p = algo.operator.params.p
    assert_array_equal(actual_p, expected_p)


# -------------------------------------------------------------------
#  BinomialPBIL
# -------------------------------------------------------------------
def test_binomial_pbil_perturb_updates_p(rng, dummy_objfunc, dummy_initializer):
    pop = Population(dummy_objfunc, np.ones((2, 2)))
    pop.fitness = np.zeros(2)

    # Expected p_hat = sum / (n * pop_size) = 2/(3*2) = 1/3
    expected_p = np.full(2, 2 / (3 * 2))

    algo = BinomialPBIL(
        initializer=dummy_initializer,
        p=0.5,
        n=3,
        noise=0,
        lr=1.0,
        random_state=rng,
    )

    algo.estimate_parameters(pop)

    actual_p = algo.operator.params.p
    assert_array_equal(actual_p, expected_p)


# -------------------------------------------------------------------
#  GaussianPBIL
# -------------------------------------------------------------------
def test_gaussian_pbil_perturb_updates_loc(rng, dummy_objfunc, dummy_initializer):
    pop = Population(dummy_objfunc, np.array([[1.0, 2.0], [3.0, 4.0]]))
    pop.fitness = np.zeros(2)

    # Expected loc = mean of original parents = [2., 3.]
    expected_loc = pop.genotype_matrix.mean(axis=0)

    algo = GaussianPBIL(
        initializer=dummy_initializer,
        loc=np.array([0.0, 0.0]),
        scale=1.0,
        lr=1.0,
        noise=0,
        random_state=rng,
    )

    algo.estimate_parameters(pop)

    actual_loc = algo.operator.params.loc
    assert_array_equal(actual_loc, expected_loc)


# -------------------------------------------------------------------
#  BernoulliUMDA
# -------------------------------------------------------------------
def test_bernoulli_umda_perturb_updates_p(rng, dummy_objfunc, dummy_initializer):
    pop = Population(dummy_objfunc, np.array([[1, 0], [1, 1]]))
    pop.fitness = np.zeros(2)

    # Expected p = mean of parents = [1., 0.5]
    expected_p = pop.genotype_matrix.mean(axis=0)

    algo = BernoulliUMDA(
        initializer=dummy_initializer,
        p=0.5,
        noise=0,
        random_state=rng,
    )

    algo.estimate_parameters(pop)

    actual_p = algo.operator.params.p
    assert_array_equal(actual_p, expected_p)


# -------------------------------------------------------------------
#  BinomialUMDA
# -------------------------------------------------------------------
def test_binomial_umda_perturb_updates_p(rng, dummy_objfunc, dummy_initializer):
    pop = Population(dummy_objfunc, np.ones((3, 2)) * 3)
    pop.fitness = np.zeros(3)

    # sum per column = 9, pop_size=3, n=5 -> 9/(5*3) = 0.6
    expected_p = np.full(2, 9 / (5 * 3))

    algo = BinomialUMDA(
        initializer=dummy_initializer,
        p=0.5,
        n=5,
        noise=0,
        random_state=rng,
    )

    algo.estimate_parameters(pop)

    actual_p = algo.operator.params.p
    assert_array_equal(actual_p, expected_p)


# -------------------------------------------------------------------
#  GaussianUMDA
# -------------------------------------------------------------------
def test_gaussian_umda_perturb_updates_loc(rng, dummy_objfunc, dummy_initializer):
    pop = Population(dummy_objfunc, np.array([[0.0, 0.0], [2.0, 2.0]]))
    pop.fitness = np.zeros(2)

    # Expected loc = mean of parents = [1., 1.]
    expected_loc = pop.genotype_matrix.mean(axis=0)

    algo = GaussianUMDA(
        initializer=dummy_initializer,
        loc=np.array([0.0, 1.0]),
        scale=1.0,
        noise=0,
        random_state=rng,
    )

    algo.estimate_parameters(pop)

    actual_loc = algo.operator.params.loc
    assert_array_equal(actual_loc, expected_loc)
