import pytest
import numpy as np
from numpy.testing import assert_array_equal

from conftest import rng, dummy_objfunc, simple_encoding, make_pop

from metaheuristic_designer.operators.factories.permutation import create_permutation_operator
from metaheuristic_designer.operator import OperatorFromLambda
from metaheuristic_designer.population import Population


# -------------------------------------------------------------------
#  Factory: method lookup and type
# -------------------------------------------------------------------
@pytest.mark.parametrize("method", ["swap", "scramble", "invert", "roll", "pmx", "order_cross"])
def test_create_permutation_operator_returns_operator(method, rng, simple_encoding):
    op = create_permutation_operator(method, encoding=simple_encoding, rng=rng)
    assert isinstance(op, OperatorFromLambda)
    assert op.name == method


def test_create_permutation_operator_default_encoding(rng):
    op = create_permutation_operator("swap", rng=rng)
    assert op.encoding is not None


def test_create_permutation_operator_invalid_method():
    with pytest.raises(KeyError):
        create_permutation_operator("nonexistent")


# -------------------------------------------------------------------
#  Integration: calling the operator modifies population
# -------------------------------------------------------------------
def test_swap_operator_modifies_genotype(rng, dummy_objfunc, simple_encoding):
    # Use an integer genotype matrix
    geno = np.array([[1, 2, 3, 4], [3, 4, 2, 1], [4, 1, 3, 2], [2, 3, 1, 4]])
    pop = Population(dummy_objfunc, geno)
    pop.fitness = np.zeros(4)

    op = create_permutation_operator("swap", encoding=simple_encoding, rng=rng)
    result = op(pop)

    assert result is pop
    # After swap, genotype should differ from original
    assert not np.array_equal(pop.genotype_matrix, geno)


def test_permutation_operator_reproducible(rng, dummy_objfunc, simple_encoding):
    geno = np.array([[1, 2, 3, 4], [3, 4, 2, 1], [4, 1, 3, 2], [2, 3, 1, 4]])
    pop1 = Population(dummy_objfunc, geno.copy())
    pop2 = Population(dummy_objfunc, geno.copy())

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    op1 = create_permutation_operator("swap", encoding=simple_encoding, rng=rng1)
    op2 = create_permutation_operator("swap", encoding=simple_encoding, rng=rng2)

    op1(pop1)
    op2(pop2)

    assert_array_equal(pop1.genotype_matrix, pop2.genotype_matrix)
