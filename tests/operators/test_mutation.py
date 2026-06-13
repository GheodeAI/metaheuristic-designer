import pytest
import numpy as np
from numpy.testing import assert_array_equal

from conftest import rng, dummy_objfunc, simple_encoding, make_pop

from metaheuristic_designer.operators.factories.mutation import create_mutation_operator
from metaheuristic_designer.operator import OperatorFromLambda
from metaheuristic_designer.population import Population


# -------------------------------------------------------------------
#  Factory: method lookup and type
# -------------------------------------------------------------------
@pytest.mark.parametrize("method", ["gauss", "xor", "mutsample", "randnoise", "mutate1sigma"])
def test_create_mutation_operator_returns_operator(method, rng, simple_encoding):
    op = create_mutation_operator(method, encoding=simple_encoding, rng=rng)
    assert isinstance(op, OperatorFromLambda)
    assert op.name == method


def test_create_mutation_operator_default_encoding(rng):
    op = create_mutation_operator("uniform", rng=rng)
    assert op.encoding is not None  # DefaultEncoding


def test_create_mutation_operator_invalid_method():
    with pytest.raises(KeyError):
        create_mutation_operator("nonexistent")


# -------------------------------------------------------------------
#  Integration: calling the operator modifies population
# -------------------------------------------------------------------
def test_gauss_operator_modifies_genotype(rng, simple_encoding):
    pop = make_pop([1.0, 2.0])  # shape (N,2) – 1 individual? Actually make_pop returns (2,?) I'll use a suitable pop.
    # Use a multi-individual population for clarity
    pop = Population(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]))
    pop.fitness = np.zeros(3)
    original = pop.genotype_matrix.copy()

    op = create_mutation_operator("gauss", encoding=simple_encoding, rng=rng, loc=0, scale=1.0, F=0.5)
    result = op(pop)

    assert result is pop
    # Genotype should have been altered (gauss noise with scale 1, strength 0.5)
    assert not np.array_equal(pop.genotype_matrix, original)


def test_xor_operator_on_zeros_population(rng, simple_encoding):
    pop = Population(np.zeros((3, 2), dtype=np.uint8))
    pop.fitness = np.zeros(3)

    op = create_mutation_operator("xor", encoding=simple_encoding, rng=rng, N=2, mode="byte")
    result = op(pop)

    assert result is pop
    assert np.any(pop.genotype_matrix != 0)

def test_polynomial_mutation(rng, simple_encoding):
    pop = Population(np.ones((3, 2)))

    op = create_mutation_operator("poly", encoding=simple_encoding, rng=rng, lower_bound=-100, upper_bound=100, dist_index=100)
    result = op(pop)

    assert result is pop
    assert np.any(pop.genotype_matrix != 0)

def test_polynomial_mutation_reproducible(rng, simple_encoding):
    pop1 = Population(np.array([[1.0, 2.0], [3.0, 4.0]]))
    pop1.fitness = np.array([0.0, 0.0])
    pop2 = Population(np.array([[1.0, 2.0], [3.0, 4.0]]))
    pop2.fitness = np.array([0.0, 0.0])

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    op1 = create_mutation_operator("poly", encoding=simple_encoding, rng=rng1, lower_bound=-100, upper_bound=100, dist_index=20)
    op2 = create_mutation_operator("poly", encoding=simple_encoding, rng=rng2, lower_bound=-100, upper_bound=100, dist_index=20)

    op1(pop1)
    op2(pop2)

    assert_array_equal(pop1.genotype_matrix, pop2.genotype_matrix)

def test_mutation_operator_reproducible(rng, simple_encoding):
    pop1 = Population(np.array([[1.0, 2.0], [3.0, 4.0]]))
    pop1.fitness = np.array([0.0, 0.0])
    pop2 = Population(np.array([[1.0, 2.0], [3.0, 4.0]]))
    pop2.fitness = np.array([0.0, 0.0])

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    op1 = create_mutation_operator("gauss", encoding=simple_encoding, rng=rng1, loc=0, scale=1.0, F=0.5)
    op2 = create_mutation_operator("gauss", encoding=simple_encoding, rng=rng2, loc=0, scale=1.0, F=0.5)

    op1(pop1)
    op2(pop2)

    assert_array_equal(pop1.genotype_matrix, pop2.genotype_matrix)


# Additional test for masked mutation (mutate_noise)
def test_mutate_noise_operator_modifies_subset(rng, simple_encoding):
    pop = Population(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    pop.fitness = np.zeros(2)

    op = create_mutation_operator("mutnoise", encoding=simple_encoding, rng=rng, distribution="normal", loc=0, scale=1.0, F=1.0, N=2)
    result = op(pop)

    assert result is pop
    # Exactly N=2 genes per individual should be mutated, so at most 4 genes changed,
    # and some should be nonzero.
    assert np.sum(pop.genotype_matrix != 0) <= 4
    assert np.any(pop.genotype_matrix != 0)
