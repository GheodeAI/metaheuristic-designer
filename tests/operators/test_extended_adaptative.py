import pytest
import numpy as np
from numpy.testing import assert_array_equal

from conftest import rng, DummyParameterExtendingEncoding, dummy_objfunc

from metaheuristic_designer.operators.extended_operator import ExtendedOperator
from metaheuristic_designer.operators.adaptive_operator import AdaptiveOperator
from metaheuristic_designer.operator import OperatorFromLambda
from metaheuristic_designer.population import Population


# ===================================================================
#  ExtendedOperator
# ===================================================================
def test_extended_operator_rejects_non_extending_encoding(rng):
    from metaheuristic_designer.encoding import DefaultEncoding

    regular_enc = DefaultEncoding()
    with pytest.raises(TypeError):
        ExtendedOperator(OperatorFromLambda(lambda p, i, rng, **kw: p, random_state=rng), {}, regular_enc)


def test_extended_operator_applies_masked_ops(rng, dummy_objfunc):
    # Encoding with solution size=2 and one param "speed" size=2 → total 4 columns
    enc = DummyParameterExtendingEncoding([("speed", 2)])
    enc.vecsize = 2

    # Base operator: adds 10 to the solution part (it receives only solution columns)
    base_op = OperatorFromLambda(lambda pop, init, rng, **kw: pop.update_genotype(pop.genotype_matrix + 10), random_state=rng)
    # Param operator: multiplies param part by 2
    param_op = OperatorFromLambda(lambda pop, init, rng, **kw: pop.update_genotype(pop.genotype_matrix * 2), random_state=rng)

    ext = ExtendedOperator(base_op, {"speed": param_op}, enc)

    # Build population with known genotype
    pop = Population(dummy_objfunc, np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]), encoding=enc)
    pop.fitness = np.zeros(2)

    result = ext.evolve(pop)

    # Expected: solution part +10, param part *2
    expected = np.array([[11.0, 12.0, 6.0, 8.0], [15.0, 16.0, 14.0, 16.0]])
    assert_array_equal(result.genotype_matrix, expected)


# ===================================================================
#  AdaptativeOperator
# ===================================================================
def test_adaptative_operator_updates_base_operator_kwargs(rng, dummy_objfunc):
    enc = DummyParameterExtendingEncoding([("speed", 2)])
    enc.vecsize = 2

    # Base operator that records kwargs and does nothing else
    recorded_kwargs = {}

    def record_kwargs(pop, init, rng, **kw):
        for k, v in kw.items():
            recorded_kwargs[k] = v
        return pop

    base_op = OperatorFromLambda(record_kwargs, random_state=rng)

    # Param operator (identity)
    param_op = OperatorFromLambda(lambda pop, init, rng, **kw: pop, random_state=rng)

    adapt = AdaptiveOperator(base_op, {"speed": param_op}, enc)

    pop = Population(dummy_objfunc, np.array([[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0]]), encoding=enc)
    pop.fitness = np.zeros(2)
    adapt.evolve(pop)

    # The speed param should have been decoded from the genotype and passed to base_op
    assert "speed" in recorded_kwargs
    assert_array_equal(recorded_kwargs["speed"], np.array([[5.0, 6.0], [7.0, 8.0]]))
