import numpy as np
from numpy.testing import assert_array_equal

from conftest import (
    rng,
    dummy_objfunc,
    make_pop,
)

from metaheuristic_designer.encoding import EncodingFromLambda
from metaheuristic_designer.objective_function import ObjectiveFromLambda
from metaheuristic_designer.initializer import InitializerFromLambda
from metaheuristic_designer.operator import OperatorFromLambda
from metaheuristic_designer.parent_selection_base import ParentSelectionFromLambda
from metaheuristic_designer.survivor_selection_base import SurvivorSelectionFromLambda
from metaheuristic_designer.constraint_handler import ConstraintHandlerFromLambda


# ===================================================================
#  EncodingFromLambda
# ===================================================================
def test_encoding_from_lambda():
    encode_called = []
    decode_called = []
    enc = EncodingFromLambda(
        encode_fn=lambda x: encode_called.append(x) or x * 2,
        decode_fn=lambda x: decode_called.append(x) or x + 10,
    )
    inp = np.array([1, 2, 3])
    assert_array_equal(enc.encode(inp), inp * 2)
    assert len(encode_called) == 1
    assert_array_equal(enc.decode(inp), inp + 10)
    assert len(decode_called) == 1


# ===================================================================
#  ObjectiveFromLambda
# ===================================================================
def test_objective_from_lambda():
    f = ObjectiveFromLambda(lambda x, **kw: float(x.sum()), dimension=3)
    result = f.objective(np.array([1.0, 2.0, 3.0]))
    assert result == 6.0


# ===================================================================
#  InitializerFromLambda
# ===================================================================
def test_initializer_from_lambda(rng):
    def my_gen(rng):
        return rng.uniform(10, 20, size=3)

    init = InitializerFromLambda(my_gen, dimension=3, pop_size=2, rng=rng)
    vec = init.generate_random()
    assert vec.shape == (3,)
    assert np.all(vec >= 10) and np.all(vec <= 20)


# ===================================================================
#  OperatorFromLambda
# ===================================================================
def test_operator_from_lambda_applies_function(rng):
    pop = make_pop([1.0, 2.0])
    original = pop.genotype_matrix.copy()

    def add_ten(p, rng, **kw):
        return p.update_genotype(p.genotype_matrix + 10)

    op = OperatorFromLambda(add_ten, rng=rng)
    result = op.evolve(pop)
    expected = original + 10
    assert_array_equal(result.genotype_matrix, expected)


# ===================================================================
#  ParentSelectionFromLambda
# ===================================================================
def test_parent_selection_from_lambda_selects_correctly(rng):
    pop = make_pop([5.0, 1.0, 3.0, 2.0])

    # Select the two best individuals
    def select_best_two(population, amount, rng):
        fitness = population.fitness
        return np.argsort(fitness)[::-1][:amount]

    sel = ParentSelectionFromLambda(select_best_two, amount=2, rng=rng)
    result = sel.select(pop, 2)
    assert len(result) == 2
    # Best fitness values are at indices 0 (5.0) and 2 (3.0)
    # The order of selected individuals is not specified, so check that the set of rows is correct
    selected_vals = result.genotype_matrix
    # The selected rows should be the ones with fitness 5.0 and 3.0
    assert any(np.array_equal(row, pop.genotype_matrix[0]) for row in selected_vals)
    assert any(np.array_equal(row, pop.genotype_matrix[2]) for row in selected_vals)


# ===================================================================
#  SurvivorSelectionFromLambda
# ===================================================================
def test_survivor_selection_from_lambda(rng):
    parents = make_pop([5.0, 1.0])
    offspring = make_pop([10.0, 2.0])

    # A lambda that always returns offspring indices [0,1] (so the offspring becomes the new population)
    def select_all_offspring(pop_fit, off_fit, rng):
        return np.array([0, 1]) + len(pop_fit)

    sel = SurvivorSelectionFromLambda(select_all_offspring, rng=rng)
    result = sel.select(parents, offspring)
    assert len(result) == 2
    assert np.all(result.genotype_matrix == offspring.genotype_matrix)


# ===================================================================
#  ConstraintHandlerFromLambda
# ===================================================================
def test_constraint_handler_from_lambda():
    handler = ConstraintHandlerFromLambda(
        repair_solution_fn=lambda x: x + 1,
        penalty_fn=lambda x: 2.0,
    )
    assert_array_equal(handler.repair_solutions(np.array([0.0])), [1.0])
    assert handler.penalty(np.array([0.0])) == 2.0


def test_constraint_handler_from_lambda_missing_both_raises():
    import pytest

    with pytest.raises(ValueError):
        ConstraintHandlerFromLambda()
