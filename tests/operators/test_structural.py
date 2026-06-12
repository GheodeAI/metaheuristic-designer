import pytest
import numpy as np
from numpy.testing import assert_array_equal

from conftest import rng, dummy_operator, dummy_objfunc, make_pop

from metaheuristic_designer.operators.composite_operator import CompositeOperator
from metaheuristic_designer.operators.branch_operator import BranchOperator
from metaheuristic_designer.operators.masked_operator import MaskedOperator
from metaheuristic_designer.operator import OperatorFromLambda


# ===================================================================
#  CompositeOperator
# ===================================================================
def test_composite_operator_applies_sequence(rng):
    # Two operators: first adds 10, second multiplies by 2
    op1 = OperatorFromLambda(lambda pop, rng, **kw: pop.update_genotype(pop.genotype_matrix + 10), name="add10", rng=rng)
    op2 = OperatorFromLambda(lambda pop, rng, **kw: pop.update_genotype(pop.genotype_matrix * 2), name="mul2", rng=rng)
    comp = CompositeOperator([op1, op2])

    pop = make_pop([0.0, 0.0])
    # manually set genotype to something known
    original = np.array([[1.0, 2.0], [3.0, 4.0]])
    pop.update_genotype(original.copy())

    result = comp.evolve(pop)
    # Expected: (original + 10) * 2
    expected = (original + 10) * 2
    assert_array_equal(result.genotype_matrix, expected)


def test_composite_operator_empty_list_does_nothing():
    comp = CompositeOperator([])
    pop = make_pop([1.0, 2.0])
    original = pop.genotype_matrix.copy()
    result = comp.evolve(pop)
    assert_array_equal(result.genotype_matrix, original)


# ===================================================================
#  BranchOperator
# ===================================================================
def test_branch_operator_random_mode(rng):
    # Two operators: one sets genotype to 0, other sets genotype to 1
    op_zero = OperatorFromLambda(lambda pop, rng, **kw: pop.update_genotype(np.zeros_like(pop.genotype_matrix)), rng=rng)
    op_one = OperatorFromLambda(lambda pop, rng, **kw: pop.update_genotype(np.ones_like(pop.genotype_matrix)), rng=rng)
    branch = BranchOperator([op_zero, op_one], method="random", rng=rng, p=0.5)

    pop = make_pop([0.0, 0.0, 0.0, 0.0])
    # initial genotype non‑zero to detect change
    pop.update_genotype(np.ones((4, 2)) * 5.0)

    result = branch.evolve(pop)
    # Each individual should be either all zeros or all ones
    for row in result.genotype_matrix:
        assert np.all(row == 0) or np.all(row == 1)


def test_branch_operator_pick_mode(rng):
    op_a = OperatorFromLambda(lambda pop, rng, **kw: pop.update_genotype(pop.genotype_matrix + 1), rng=rng)
    op_b = OperatorFromLambda(lambda pop, rng, **kw: pop.update_genotype(pop.genotype_matrix - 1), rng=rng)
    branch = BranchOperator([op_a, op_b], method="pick", rng=rng, idx=1)

    pop = make_pop([0.0, 0.0])
    pop.update_genotype(np.array([[10.0, 10.0], [20.0, 20.0]]))
    result = branch.evolve(pop)
    # idx=1 always picks operator B (subtract 1)
    expected = pop.genotype_matrix - 1
    assert_array_equal(result.genotype_matrix, expected)


# ===================================================================
#  MaskedOperator
# ===================================================================
def test_masked_operator_applies_different_ops_per_column(rng):
    # Mask: first column (0) gets op_zero, second column (1) gets op_one
    mask = np.array([0, 1])
    op_zero = OperatorFromLambda(lambda pop, rng, **kw: pop.update_genotype(np.zeros_like(pop.genotype_matrix)), rng=rng)
    op_one = OperatorFromLambda(lambda pop, rng, **kw: pop.update_genotype(np.ones_like(pop.genotype_matrix)), rng=rng)
    masked = MaskedOperator([op_zero, op_one], mask=mask)

    pop = make_pop([0.0, 0.0])
    pop.update_genotype(np.array([[7.0, 8.0], [9.0, 10.0]]))
    result = masked.evolve(pop)

    # Column 0 -> 0, column 1 -> 1
    expected = np.array([[0.0, 1.0], [0.0, 1.0]])
    assert_array_equal(result.genotype_matrix, expected)
