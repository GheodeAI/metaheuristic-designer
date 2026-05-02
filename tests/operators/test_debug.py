import pytest
import numpy as np
from numpy.testing import assert_array_equal

from conftest import rng, dummy_objfunc, simple_encoding, make_pop

from metaheuristic_designer.operators.factories.debug import create_debug_operator
from metaheuristic_designer.operator import OperatorFromLambda


@pytest.mark.parametrize("method", ["debug", "zeros", "ones"])
def test_create_debug_operator_returns_operator(method, rng, simple_encoding):
    op = create_debug_operator(method, encoding=simple_encoding, random_state=rng)
    assert isinstance(op, OperatorFromLambda)
    assert op.name == method


def test_create_debug_operator_invalid_method():
    with pytest.raises(KeyError):
        create_debug_operator("not_a_debug_op")


def test_zeros_operator_fills_with_zero(rng, dummy_objfunc, simple_encoding):
    pop = make_pop([1.0, 2.0], dummy_objfunc)
    op = create_debug_operator("zeros", encoding=simple_encoding, random_state=rng)
    result = op(pop)
    assert result is pop
    assert_array_equal(pop.genotype_matrix, np.zeros_like(pop.genotype_matrix))


def test_ones_operator_fills_with_one(rng, dummy_objfunc, simple_encoding):
    pop = make_pop([5.0, 5.0], dummy_objfunc)
    op = create_debug_operator("ones", encoding=simple_encoding, random_state=rng)
    result = op(pop)
    assert result is pop
    assert_array_equal(pop.genotype_matrix, np.ones_like(pop.genotype_matrix))


def test_debug_operator_reproducible(rng, dummy_objfunc, simple_encoding):
    pop1 = make_pop([1.0, 2.0], dummy_objfunc)
    pop2 = make_pop([1.0, 2.0], dummy_objfunc)
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    op1 = create_debug_operator("zeros", encoding=simple_encoding, random_state=rng1)
    op2 = create_debug_operator("zeros", encoding=simple_encoding, random_state=rng2)
    op1(pop1)
    op2(pop2)
    assert_array_equal(pop1.genotype_matrix, pop2.genotype_matrix)