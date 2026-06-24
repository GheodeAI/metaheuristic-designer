import pytest
import numpy as np
from numpy.testing import assert_array_equal

from conftest import rng, dummy_objfunc, simple_encoding, pso_population

from metaheuristic_designer.operators.factories.swarm import create_swarm_operator
from metaheuristic_designer.operator import OperatorFromLambda
from metaheuristic_designer.encodings import ParameterExtendingEncoding


# ===================================================================
#  Factory
# ===================================================================
def test_create_swarm_operator_returns_operator(rng, pso_population):
    op = create_swarm_operator("pso", encoding=pso_population.encoding, rng=rng)
    assert isinstance(op, OperatorFromLambda)
    assert op.name == "pso"


def test_create_swarm_operator_invalid_method(rng, pso_population):
    with pytest.raises(KeyError):
        create_swarm_operator("swarm_that_does_not_exist", encoding=pso_population.encoding)


# ===================================================================
#  Integration: pso_operator_wrapper (via factory)
# ===================================================================
def test_pso_operator_wrapper_updates_genotype(rng, pso_population):
    original_geno = pso_population.genotype_matrix.copy()
    op = create_swarm_operator("pso", encoding=pso_population.encoding, rng=rng, w=0.7, c1=1.5, c2=1.5)
    result = op(pso_population)

    # The population itself is returned (same object)
    assert result is pso_population
    # The genotype should have changed
    assert not np.array_equal(pso_population.genotype_matrix, original_geno)
    # Speed component should be updated (last columns)
    speed = pso_population.encoding.decode_params(pso_population.genotype_matrix)["speed"]
    assert speed.shape[1] == 2


def test_pso_operator_wrapper_reproducible():
    # We'll create two identical populations with same encoding and call the wrapper.
    from metaheuristic_designer.encodings import PSOEncoding
    from metaheuristic_designer.encoding import DefaultEncoding
    from metaheuristic_designer.population import Population

    enc = PSOEncoding(dimension=2, base_encoding=DefaultEncoding())
    geno = np.array([[1.0, 2.0, 0.1, 0.2], [3.0, 4.0, 0.3, 0.4]])
    pop1 = Population(geno.copy(), encoding=enc)
    pop2 = Population(geno.copy(), encoding=enc)
    for pop in (pop1, pop2):
        pop.fitness = np.array([0.5, 0.8])
        pop.historical_best_matrix = geno.copy()
        pop.historical_best_fitness = np.array([0.5, 0.8])
        pop.best = np.array([3.0, 4.0, 0.3, 0.4])
        pop.best_fitness = 0.8

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    op1 = create_swarm_operator("pso", encoding=enc, rng=rng1, w=0.7, c1=1.5, c2=1.5)
    op2 = create_swarm_operator("pso", encoding=enc, rng=rng2, w=0.7, c1=1.5, c2=1.5)
    op1(pop1)
    op2(pop2)

    assert_array_equal(pop1.genotype_matrix, pop2.genotype_matrix)
