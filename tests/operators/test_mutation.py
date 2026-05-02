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
@pytest.mark.parametrize("method", [
    "gauss", "xor", "mutsample", "randnoise", "mutate1sigma"
])
def test_create_mutation_operator_returns_operator(method, rng, simple_encoding):
    op = create_mutation_operator(method, encoding=simple_encoding, random_state=rng)
    assert isinstance(op, OperatorFromLambda)
    assert op.name == method


def test_create_mutation_operator_default_encoding(rng):
    op = create_mutation_operator("uniform", random_state=rng)
    assert op.encoding is not None    # DefaultEncoding


def test_create_mutation_operator_invalid_method():
    with pytest.raises(KeyError):
        create_mutation_operator("nonexistent")


# -------------------------------------------------------------------
#  Integration: calling the operator modifies population
# -------------------------------------------------------------------
def test_gauss_operator_modifies_genotype(rng, dummy_objfunc, simple_encoding):
    pop = make_pop([0.0, 0.0], dummy_objfunc)
    original = pop.genotype_matrix.copy()

    op = create_mutation_operator("gauss", encoding=simple_encoding, random_state=rng,
                                  N=2, distrib="gauss", loc=0, scale=1)
    result = op(pop)

    assert result is pop
    # Genotype should have been altered (gauss noise with scale 1)
    assert not np.array_equal(pop.genotype_matrix, original)


def test_xor_operator_on_zeros_population(rng, dummy_objfunc, simple_encoding):
    pop = Population(dummy_objfunc, np.zeros((3, 2), dtype=np.uint8))
    pop.fitness = np.zeros(3)

    op = create_mutation_operator("xor", encoding=simple_encoding, random_state=rng,
                                  N=2, BinRep="byte")
    result = op(pop)

    assert result is pop
    assert np.any(pop.genotype_matrix != 0)


def test_mutation_operator_reproducible(rng, dummy_objfunc, simple_encoding):
    pop1 = make_pop([1.0, 2.0], dummy_objfunc)
    pop2 = make_pop([1.0, 2.0], dummy_objfunc)

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    op1 = create_mutation_operator("gauss", encoding=simple_encoding, random_state=rng1,
                                   N=2, distrib="gauss", loc=0, scale=1)
    op2 = create_mutation_operator("gauss", encoding=simple_encoding, random_state=rng2,
                                   N=2, distrib="gauss", loc=0, scale=1)

    op1(pop1)
    op2(pop2)

    assert_array_equal(pop1.genotype_matrix, pop2.genotype_matrix)