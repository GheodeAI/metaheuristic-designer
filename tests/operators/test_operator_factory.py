import pytest
import numpy as np
from conftest import rng, simple_encoding

from metaheuristic_designer.operators import create_operator, add_operator_entry
from metaheuristic_designer.operator import OperatorFromLambda, NullOperator
from metaheuristic_designer.operators.BO_operator import BOOperator


# -------------------------------------------------------------------
#  Direct registry lookups
# -------------------------------------------------------------------
def test_create_operator_registry_lookup(rng, simple_encoding):
    op = create_operator("mutation.gauss", encoding=simple_encoding, rng=rng)
    assert isinstance(op, OperatorFromLambda)
    assert op.name == "mutation.gauss"


def test_create_operator_crossover(rng, simple_encoding):
    op = create_operator("crossover.one_point", encoding=simple_encoding, rng=rng)
    assert isinstance(op, OperatorFromLambda)


def test_create_operator_invalid_registry():
    with pytest.raises(ValueError):
        create_operator("invalid.blah")


def test_create_operator_invalid_method_in_registry():
    with pytest.raises(ValueError):
        create_operator("mutation.nonexistent")


# -------------------------------------------------------------------
#  Unqualified name search across registries
# -------------------------------------------------------------------
def test_create_operator_unqualified_gauss(rng, simple_encoding):
    op = create_operator("gauss", encoding=simple_encoding, rng=rng)
    assert isinstance(op, OperatorFromLambda)
    assert op.name == "gauss"


def test_create_operator_unqualified_swap(rng, simple_encoding):
    op = create_operator("swap", encoding=simple_encoding, rng=rng)
    assert isinstance(op, OperatorFromLambda)


# -------------------------------------------------------------------
#  Special aliases
# -------------------------------------------------------------------
def test_create_operator_null_alias():
    op = create_operator("nothing")
    assert isinstance(op, NullOperator)


def test_create_operator_bo_alias(rng, simple_encoding, dummy_objfunc):
    op = create_operator("bo", objfunc=dummy_objfunc, encoding=simple_encoding, rng=rng)
    assert isinstance(op, BOOperator)
    assert op.name == "bo"


# -------------------------------------------------------------------
#  Name override
# -------------------------------------------------------------------
def test_create_operator_custom_name(rng, simple_encoding):
    op = create_operator("gauss", name="my_gauss", encoding=simple_encoding, rng=rng)
    assert op.name == "my_gauss"


# -------------------------------------------------------------------
#  Invalid method overall
# -------------------------------------------------------------------
def test_create_operator_completely_unknown():
    with pytest.raises(ValueError):
        create_operator("this_method_does_not_exist")


# -------------------------------------------------------------------
#  add_operator_entry
# -------------------------------------------------------------------
def test_add_and_retrieve_custom_operator(rng, simple_encoding):
    # Define a dummy function with the required signature
    def dummy_op(pop, init, rng, **kw):
        return pop

    add_operator_entry(dummy_op, "my_custom_op", "custom")
    # Now retrieve it
    op = create_operator("custom.my_custom_op", encoding=simple_encoding, rng=rng)
    assert isinstance(op, OperatorFromLambda)
    # Clean up so other tests aren't affected
    from metaheuristic_designer.operators.factories.generic import all_ops_map

    del all_ops_map["custom"]["my_custom_op"]
