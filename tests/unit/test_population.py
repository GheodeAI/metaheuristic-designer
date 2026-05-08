"""
Unit tests for Population.

Contract being verified:
- Construction initialises fitness to -inf for every individual.
- calculate_fitness populates fitness and marks individuals as evaluated.
- best_individual / best_solution return the individual with the highest fitness.
- take_selection returns a Population restricted to the chosen rows.
- join_populations produces a Population whose size is the sum of both.
- Population.decode returns a list of the decoded solutions.
"""

import numpy as np
import pytest

from metaheuristic_designer.population import Population
from metaheuristic_designer.benchmarks.benchmark_funcs import MaxOnes, Sphere


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_population_shape_attributes(onemax_func):
    geno = np.zeros((4, 8))
    pop = Population(onemax_func, geno)
    assert pop.pop_size == 4
    assert pop.vec_size == 8


def test_population_initial_fitness_is_negative_inf(onemax_func):
    geno = np.zeros((3, 8))
    pop = Population(onemax_func, geno)
    assert np.all(pop.fitness == -np.inf)


def test_population_initial_fitness_calculated_is_false(onemax_func):
    geno = np.zeros((3, 8))
    pop = Population(onemax_func, geno)
    assert not np.any(pop.fitness_calculated)


# ---------------------------------------------------------------------------
# Fitness calculation
# ---------------------------------------------------------------------------

def test_calculate_fitness_marks_all_individuals(evaluated_population):
    assert np.all(evaluated_population.fitness_calculated)


def test_calculate_fitness_values_match_objective(onemax_func):
    # MaxOnes fitness = sum of ones; with mode="max" and no penalty, fitness == objective
    geno = np.array([[1, 1, 0, 0, 0, 0, 0, 0],   # sum = 2
                     [1, 1, 1, 1, 0, 0, 0, 0]], dtype=float)  # sum = 4
    pop = Population(onemax_func, geno)
    pop.calculate_fitness()
    assert pop.fitness[0] == pytest.approx(2.0)
    assert pop.fitness[1] == pytest.approx(4.0)


def test_calculate_fitness_identifies_best(onemax_func):
    geno = np.array([[1, 1, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1, 1, 1]], dtype=float)
    pop = Population(onemax_func, geno)
    pop.calculate_fitness()
    _, best_fit = pop.best_individual()
    assert best_fit == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# Best individual / solution
# ---------------------------------------------------------------------------

def test_best_individual_returns_highest_fitness(onemax_func):
    geno = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 0, 0, 0]], dtype=float)
    pop = Population(onemax_func, geno)
    pop.calculate_fitness()
    vec, fit = pop.best_individual()
    assert fit == pytest.approx(5.0)
    assert np.array_equal(vec, geno[2])


def test_best_solution_returns_decoded_solution(onemax_func):
    geno = np.array([[1, 1, 1, 1, 1, 1, 1, 1]], dtype=float)
    pop = Population(onemax_func, geno)
    pop.calculate_fitness()
    sol, fit = pop.best_solution()
    assert fit == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# take_selection
# ---------------------------------------------------------------------------

def test_take_selection_returns_correct_rows(evaluated_population):
    idx = np.array([0, 2, 4])
    sub = evaluated_population.take_selection(idx)
    assert sub.pop_size == 3
    assert np.array_equal(sub.genotype_matrix, evaluated_population.genotype_matrix[idx])


def test_take_selection_preserves_fitness(evaluated_population):
    idx = np.array([1, 3])
    sub = evaluated_population.take_selection(idx)
    np.testing.assert_array_equal(sub.fitness, evaluated_population.fitness[idx])


# ---------------------------------------------------------------------------
# join_populations
# ---------------------------------------------------------------------------

def test_join_populations_size(evaluated_population):
    joined = Population.join_populations(evaluated_population, evaluated_population)
    assert joined.pop_size == evaluated_population.pop_size * 2


def test_join_populations_genotype(evaluated_population):
    joined = Population.join_populations(evaluated_population, evaluated_population)
    expected = np.concatenate([evaluated_population.genotype_matrix,
                               evaluated_population.genotype_matrix], axis=0)
    np.testing.assert_array_equal(joined.genotype_matrix, expected)


# ---------------------------------------------------------------------------
# Iteration and length
# ---------------------------------------------------------------------------

def test_population_len(evaluated_population):
    assert len(evaluated_population) == evaluated_population.pop_size


def test_population_iteration_yields_rows(evaluated_population):
    rows = list(evaluated_population)
    assert len(rows) == evaluated_population.pop_size
    for i, row in enumerate(rows):
        np.testing.assert_array_equal(row, evaluated_population.genotype_matrix[i])


# ---------------------------------------------------------------------------
# update_genotype
# ---------------------------------------------------------------------------

def test_update_genotype_replaces_matrix(evaluated_population, onemax_func):
    new_geno = np.ones_like(evaluated_population.genotype_matrix)
    evaluated_population.update_genotype(new_geno)
    np.testing.assert_array_equal(evaluated_population.genotype_matrix, new_geno)


def test_update_genotype_from_population(evaluated_population, onemax_func):
    new_geno = np.zeros_like(evaluated_population.genotype_matrix)
    other_pop = Population(onemax_func, new_geno)
    evaluated_population.update_genotype(other_pop)
    np.testing.assert_array_equal(evaluated_population.genotype_matrix, new_geno)


def test_update_genotype_wrong_dimension_raises(evaluated_population):
    bad_geno = np.ones((evaluated_population.pop_size, evaluated_population.vec_size + 1))
    with pytest.raises(ValueError):
        evaluated_population.update_genotype(bad_geno)


# ---------------------------------------------------------------------------
# apply_selection
# ---------------------------------------------------------------------------

def test_apply_selection_updates_rows(evaluated_population, onemax_func):
    idx = np.array([0, 1])
    replacement_geno = np.ones((2, evaluated_population.vec_size))
    replacement_pop = Population(onemax_func, replacement_geno)
    result = evaluated_population.apply_selection(replacement_pop, idx)
    np.testing.assert_array_equal(result.genotype_matrix[idx], replacement_geno)


# ---------------------------------------------------------------------------
# take_slice / apply_slice
# ---------------------------------------------------------------------------

def test_take_slice_column_subset(evaluated_population):
    mask = np.array([0, 2, 4])
    sliced = evaluated_population.take_slice(mask)
    assert sliced.genotype_matrix.shape[1] == 3
    np.testing.assert_array_equal(sliced.genotype_matrix,
                                  evaluated_population.genotype_matrix[:, mask])


def test_apply_slice_updates_columns(evaluated_population, onemax_func):
    mask = np.array([0, 1])
    sliced_geno = np.ones((evaluated_population.pop_size, 2))
    sliced_pop = Population(onemax_func, sliced_geno)
    evaluated_population.apply_slice(sliced_pop, mask)
    np.testing.assert_array_equal(evaluated_population.genotype_matrix[:, mask], sliced_geno)


# ---------------------------------------------------------------------------
# join (instance method – mutates self, appending the other population)
# ---------------------------------------------------------------------------

def test_join_instance_method_size(onemax_func):
    """join() appends other_pop's individuals to self (mutates self in-place)."""
    rng = np.random.default_rng(0)
    geno_a = rng.integers(0, 2, size=(4, 8)).astype(float)
    geno_b = rng.integers(0, 2, size=(3, 8)).astype(float)
    pop_a = Population(onemax_func, geno_a)
    pop_b = Population(onemax_func, geno_b)
    pop_a.join(pop_b)
    assert pop_a.pop_size == 7


# ---------------------------------------------------------------------------
# Selection from an evaluated population via indices
# ---------------------------------------------------------------------------

def test_take_selection_bool_mask(evaluated_population):
    mask = np.array([True, False, True, False, True, False])
    sub = evaluated_population.take_selection(mask)
    assert sub.pop_size == 3


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------

def test_population_repr_contains_objfunc_name(evaluated_population):
    r = repr(evaluated_population)
    assert "Population{" in r


# ---------------------------------------------------------------------------
# sort_by_fitness
# ---------------------------------------------------------------------------

def test_sort_population_ascending_order(evaluated_population):
    evaluated_population.sort_population()
    assert np.all(evaluated_population.fitness[:-1] <= evaluated_population.fitness[1:])


def test_sort_population_preserves_size(evaluated_population):
    n = evaluated_population.pop_size
    evaluated_population.sort_population()
    assert evaluated_population.pop_size == n


# ---------------------------------------------------------------------------
# repeat
# ---------------------------------------------------------------------------

def test_repeat_doubles_population(evaluated_population):
    n = evaluated_population.pop_size
    repeated = evaluated_population.repeat(amount=2)
    assert repeated.pop_size == n * 2


def test_repeat_triples_population(evaluated_population):
    n = evaluated_population.pop_size
    repeated = evaluated_population.repeat(amount=3)
    assert repeated.pop_size == n * 3


def test_repeat_keeps_fitness(evaluated_population):
    repeated = evaluated_population.repeat(amount=2)
    n = evaluated_population.pop_size
    np.testing.assert_array_equal(repeated.fitness[:n], evaluated_population.fitness)
    np.testing.assert_array_equal(repeated.fitness[n:], evaluated_population.fitness)


# ---------------------------------------------------------------------------
# update_best_from_parents
# ---------------------------------------------------------------------------

def test_update_best_from_parents_promotes_better_parent(onemax_func):
    """If the parent has a better best, the child population adopts it."""
    rng = np.random.default_rng(0)
    geno_parent = rng.integers(0, 2, size=(4, 8)).astype(float)
    geno_child = np.zeros((4, 8), dtype=float)

    parent_pop = Population(onemax_func, geno_parent)
    parent_pop.calculate_fitness()

    child_pop = Population(onemax_func, geno_child)
    child_pop.calculate_fitness()

    old_best = child_pop.best_fitness
    child_pop.update_best_from_parents(parent_pop)

    if parent_pop.best_fitness > old_best:
        assert child_pop.best_fitness == parent_pop.best_fitness


def test_update_best_from_parents_keeps_own_best_if_better(onemax_func):
    """If the child has a better best than parent, it stays unchanged."""
    geno_parent = np.zeros((3, 8), dtype=float)
    geno_child = np.ones((3, 8), dtype=float)  # all-ones → best possible

    parent_pop = Population(onemax_func, geno_parent)
    parent_pop.calculate_fitness()

    child_pop = Population(onemax_func, geno_child)
    child_pop.calculate_fitness()

    original_best = child_pop.best_fitness
    child_pop.update_best_from_parents(parent_pop)

    assert child_pop.best_fitness == pytest.approx(original_best)


# ---------------------------------------------------------------------------
# step
# ---------------------------------------------------------------------------

def test_step_updates_best(onemax_func):
    geno = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1, 1, 1]], dtype=float)
    pop = Population(onemax_func, geno)
    pop.calculate_fitness()
    pop.step()
    assert pop.best_fitness == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# decode, decode_params, encode
# ---------------------------------------------------------------------------

def test_population_decode_returns_array(evaluated_population):
    decoded = evaluated_population.decode()
    assert decoded is not None
    assert hasattr(decoded, "__len__")


def test_population_decode_params_with_default_encoding_returns_none(evaluated_population):
    """Default encoding has no params → decode_params returns None."""
    result = evaluated_population.decode_params()
    assert result is None


def test_population_encode_returns_matrix(evaluated_population):
    encoded = evaluated_population.encode()
    assert encoded is not None
    assert encoded.shape == evaluated_population.genotype_matrix.shape


# ---------------------------------------------------------------------------
# get_state
# ---------------------------------------------------------------------------

def test_get_state_returns_dict(evaluated_population):
    state = evaluated_population.get_state()
    assert isinstance(state, dict)
    assert "genotype_matrix" in state
    assert "fitness" in state
    assert "best_fitness" in state


# ---------------------------------------------------------------------------
# debug_repr
# ---------------------------------------------------------------------------

def test_debug_repr_returns_string(evaluated_population):
    s = evaluated_population.debug_repr()
    assert isinstance(s, str)
    assert "Population(" in s


@pytest.mark.xfail(
    reason="BUG: debug_repr crashes with ValueError on empty population (zero-size array reduction). "
           "See ERRORES.md."
)
def test_debug_repr_empty_population(onemax_func):
    """debug_repr handles an empty genotype matrix."""
    geno = np.zeros((0, 8), dtype=float)
    pop = Population(onemax_func, geno)
    s = pop.debug_repr()
    assert isinstance(s, str)


# ---------------------------------------------------------------------------
# update_genotype with size change
# ---------------------------------------------------------------------------

def test_update_genotype_different_row_count_resets_fitness(evaluated_population, onemax_func):
    """Updating genotype with different number of rows resets fitness tracking."""
    new_geno = np.ones((2, evaluated_population.vec_size), dtype=float)  # fewer rows
    evaluated_population.update_genotype(new_geno)
    assert evaluated_population.pop_size == 2
    assert np.all(evaluated_population.fitness == -np.inf)


# ---------------------------------------------------------------------------
# apply_selection best update path
# ---------------------------------------------------------------------------

def test_apply_selection_updates_best_when_replacement_is_better(onemax_func):
    """apply_selection promotes the best from the replacement population."""
    geno_orig = np.zeros((4, 8), dtype=float)
    orig_pop = Population(onemax_func, geno_orig)
    orig_pop.calculate_fitness()

    geno_repl = np.ones((2, 8), dtype=float)  # better solutions
    repl_pop = Population(onemax_func, geno_repl)
    repl_pop.calculate_fitness()

    idx = np.array([0, 1])
    result = orig_pop.apply_selection(repl_pop, idx)
    assert result.best_fitness == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# join_populations: best selection from the better population
# ---------------------------------------------------------------------------

def test_join_populations_best_comes_from_better_subpop(onemax_func):
    geno_a = np.zeros((3, 8), dtype=float)
    geno_b = np.ones((3, 8), dtype=float)

    pop_a = Population(onemax_func, geno_a)
    pop_a.calculate_fitness()

    pop_b = Population(onemax_func, geno_b)
    pop_b.calculate_fitness()

    joined = Population.join_populations(pop_a, pop_b)
    assert joined.best_fitness == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# apply_slice updates best when slice has better solution
# ---------------------------------------------------------------------------

def test_apply_slice_best_update_when_slice_better(evaluated_population, onemax_func):
    mask = np.array([0, 1])
    sliced_geno = np.ones((evaluated_population.pop_size, 2), dtype=float)
    sliced_pop = Population(onemax_func, sliced_geno)
    sliced_pop.calculate_fitness()

    old_best = evaluated_population.best_fitness
    evaluated_population.apply_slice(sliced_pop, mask)
    # The apply_slice should update best if slice has higher best_fitness
    if sliced_pop.best_fitness > old_best:
        assert evaluated_population.best_fitness == sliced_pop.best_fitness
