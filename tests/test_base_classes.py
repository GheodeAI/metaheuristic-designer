import numpy as np
from numpy.testing import assert_array_equal

from conftest import (
    dummy_objfunc,
    dummy_initializer,
    rng,
    make_pop,
)

from metaheuristic_designer.constraint_handler import NullConstraint
from metaheuristic_designer.operator import NullOperator
from metaheuristic_designer.parent_selection_base import NullParentSelection
from metaheuristic_designer.survivor_selection_base import NullSurvivorSelection
from metaheuristic_designer.encoding import DefaultEncoding
from metaheuristic_designer.initializer import Initializer, InitializerFromLambda
from metaheuristic_designer.schedulable_parameter import SchedulableParameter


# ===================================================================
#  NullConstraint
# ===================================================================
def test_null_constraint_repair_does_nothing():
    handler = NullConstraint()
    orig = np.array([[1.0, -5.0]])
    repaired = handler.repair_solutions(orig)
    assert_array_equal(repaired, orig)
    assert repaired is not orig


def test_null_constraint_penalty_zero():
    handler = NullConstraint()
    assert_array_equal(handler.penalty(np.array([100.0, -200.0])), 0)


# ===================================================================
#  NullOperator
# ===================================================================
def test_null_operator_returns_copy():
    pop = make_pop([1.0, 2.0])
    op = NullOperator()
    result = op.evolve(pop)
    # result should be a copy, not the same object
    assert result is not pop
    assert_array_equal(result.genotype_matrix, pop.genotype_matrix)


# ===================================================================
#  NullParentSelection
# ===================================================================
def test_null_parent_selection_returns_same_population():
    pop = make_pop([5.0, 1.0])
    sel = NullParentSelection()
    result = sel.select(pop)
    np.testing.assert_allclose(result.genotype_matrix, pop.genotype_matrix)


# ===================================================================
#  NullSurvivorSelection
# ===================================================================
def test_null_survivor_selection_returns_offspring():
    parents = make_pop([10.0, 1.0])
    offspring = make_pop([5.0, 5.0])
    sel = NullSurvivorSelection()
    result = sel.select(parents, offspring)
    # NullSurvivorSelection returns the offspring
    assert result is offspring


# ===================================================================
#  DefaultEncoding
# ===================================================================
def test_default_encoding_encode_decode_identity():
    enc = DefaultEncoding()
    arr = np.array([1, 2, 3])
    assert_array_equal(enc.encode(arr), arr)
    assert_array_equal(enc.decode(arr), arr)


# ===================================================================
#  SchedulableParameter (abstract, placeholder test)
# ===================================================================
def test_schedulable_parameter_call_calls_evaluate():
    # Minimal concrete subclass for testing
    class TestParam(SchedulableParameter):
        def evaluate(self, progress):
            return progress * 2

    param = TestParam(rng=42)
    assert param(0.5) == 1.0
