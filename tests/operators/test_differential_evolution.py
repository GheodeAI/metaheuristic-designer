import pytest
import numpy as np
from numpy.testing import assert_array_equal

from conftest import rng, dummy_objfunc, simple_encoding, make_pop

from metaheuristic_designer.operators.differential_evolution_operator import create_differential_evolution_operator
from metaheuristic_designer.operator import OperatorFromLambda


@pytest.mark.parametrize("method", ["de/rand/1", "de/best/1", "de/current-to-rand/1"])
def test_create_de_operator_returns_operator(method, rng, simple_encoding):
    op = create_differential_evolution_operator(method, encoding=simple_encoding, random_state=rng)
    assert isinstance(op, OperatorFromLambda)
    assert op.name == method


def test_create_de_operator_default_encoding(rng):
    op = create_differential_evolution_operator("de.rand.1", random_state=rng)
    assert op.encoding is not None


def test_create_de_operator_invalid_method():
    with pytest.raises(KeyError):
        create_differential_evolution_operator("invalid/de")


def test_de_rand1_modifies_population(rng, dummy_objfunc, simple_encoding):
    pop = make_pop([0.0, 0.0], dummy_objfunc)
    original = pop.genotype_matrix.copy()

    op = create_differential_evolution_operator("de/rand/1", encoding=simple_encoding, random_state=rng)
    result = op(pop)

    assert result is pop
    assert not np.array_equal(pop.genotype_matrix, original)