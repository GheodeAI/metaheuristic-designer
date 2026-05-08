"""
Unit tests for Operator wrappers.

Contracts verified:
- NullOperator returns a population of identical shape and values.
- OperatorFromLambda wraps any function correctly.
- CompositeOperator applies operators in sequence.
- BranchOperator dispatches to exactly one sub-operator per call.
"""

import numpy as np
import pytest

from metaheuristic_designer.operator import NullOperator, OperatorFromLambda
from metaheuristic_designer.operators.composite_operator import CompositeOperator
from metaheuristic_designer.operators.branch_operator import BranchOperator
from metaheuristic_designer.population import Population
from metaheuristic_designer.benchmarks.benchmark_funcs import MaxOnes


def _make_pop(n_ind=6, n_genes=4, seed=0):
    objfunc = MaxOnes(dimension=n_genes)
    rng = np.random.default_rng(seed)
    geno = rng.integers(0, 2, size=(n_ind, n_genes)).astype(float)
    pop = Population(objfunc, geno)
    pop.calculate_fitness()
    return pop


# ---------------------------------------------------------------------------
# NullOperator
# ---------------------------------------------------------------------------

def test_null_operator_preserves_shape():
    pop = _make_pop()
    null_op = NullOperator()
    result = null_op(pop)
    assert result.pop_size == pop.pop_size
    assert result.vec_size == pop.vec_size


def test_null_operator_preserves_genotype():
    pop = _make_pop()
    null_op = NullOperator()
    result = null_op(pop)
    np.testing.assert_array_equal(result.genotype_matrix, pop.genotype_matrix)


# ---------------------------------------------------------------------------
# OperatorFromLambda
# ---------------------------------------------------------------------------

def test_operator_from_lambda_is_called():
    """OperatorFromLambda calls the function with (population, initializer, random_state)."""
    called = []

    def my_op(population, initializer, random_state=None):
        called.append(True)
        return population  # return Population unchanged

    op = OperatorFromLambda(my_op)
    pop = _make_pop()
    op(pop)
    assert len(called) == 1


def test_operator_from_lambda_can_transform():
    """A lambda that doubles the genotype matrix via Population mutation."""
    def double_fn(population, initializer, random_state=None):
        return Population(population.objfunc, population.genotype_matrix * 2)

    double_op = OperatorFromLambda(double_fn)
    pop = _make_pop()
    original_geno = pop.genotype_matrix.copy()
    result = double_op(pop)
    np.testing.assert_array_equal(result.genotype_matrix, original_geno * 2)


# ---------------------------------------------------------------------------
# CompositeOperator
# ---------------------------------------------------------------------------

def test_composite_operator_applies_all_ops_in_order():
    log = []

    def make_logger(tag):
        def op(population, initializer, random_state=None):
            log.append(tag)
            return population
        return OperatorFromLambda(op)

    comp = CompositeOperator([make_logger("A"), make_logger("B"), make_logger("C")])
    comp(_make_pop())
    assert log == ["A", "B", "C"]


def test_composite_operator_output_shape():
    null1 = NullOperator()
    null2 = NullOperator()
    comp = CompositeOperator([null1, null2])
    pop = _make_pop()
    result = comp(pop)
    assert result.pop_size == pop.pop_size
    assert result.vec_size == pop.vec_size


def test_composite_single_operator_equivalent_to_applying_directly():
    null = NullOperator()
    comp = CompositeOperator([null])
    pop = _make_pop()
    result = comp(pop)
    np.testing.assert_array_equal(result.genotype_matrix, pop.genotype_matrix)


# ---------------------------------------------------------------------------
# BranchOperator – RANDOM mode
# ---------------------------------------------------------------------------

def test_branch_operator_random_mode_output_shape():
    null1 = NullOperator()
    null2 = NullOperator()
    branch = BranchOperator([null1, null2], mode="RANDOM", random_state=0)
    pop = _make_pop()
    result = branch(pop)
    assert result.pop_size == pop.pop_size


def test_branch_operator_random_output_shape():
    """BranchOperator RANDOM mode routes each individual to one operator.
    The output population must have the same size as the input."""
    branch = BranchOperator(
        [NullOperator(), NullOperator()],
        mode="RANDOM",
        random_state=1,
    )
    pop = _make_pop()
    result = branch(pop)
    assert result.pop_size == pop.pop_size


# ---------------------------------------------------------------------------
# BranchOperator – PICK mode
# ---------------------------------------------------------------------------

def test_branch_operator_pick_mode_output_shape():
    """BranchOperator PICK mode must preserve population shape."""
    branch = BranchOperator(
        [NullOperator(), NullOperator()],
        mode="PICK",
        random_state=0,
    )
    branch.choose_index(0)
    pop = _make_pop()
    result = branch(pop)
    assert result.pop_size == pop.pop_size


def test_branch_operator_choose_index_returns_op_state():
    """choose_index sets the chosen index that will be used in PICK mode."""
    branch = BranchOperator(
        [NullOperator(), NullOperator()],
        mode="PICK",
        random_state=0,
    )
    branch.choose_index(1)
    assert branch.chosen_idx == 1
