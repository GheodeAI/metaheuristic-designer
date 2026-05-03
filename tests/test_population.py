import pytest
from copy import copy
import numpy as np

# Import the class to test
from metaheuristic_designer.population import Population

# conftest fixtures are automatically available

# Import shared constants from conftest
from conftest import SMALL_GENOTYPE, LARGE_GENOTYPE

# ---------------------------------------------------------------
# Initialisation and basic properties
# ---------------------------------------------------------------


@pytest.mark.parametrize(
    "genotype, expected_shape",
    [
        (np.array([[1, 2], [3, 4]]), (2, 2)),
        (np.array([[5]]), (1, 1)),
        (np.zeros((0, 2)), (0, 2)),
    ],
)
def test_initialisation_defaults(genotype, expected_shape, dummy_objfunc):
    pop = Population(dummy_objfunc, genotype)
    assert pop.pop_size == expected_shape[0]
    assert pop.vec_size == expected_shape[1]
    assert pop.fitness == pytest.approx(np.full(expected_shape[0], -np.inf))
    assert np.all(pop.fitness_calculated == 0)
    assert pop.best is None
    assert pop.best_fitness is None
    np.testing.assert_array_equal(pop.historical_best_matrix, genotype)
    np.testing.assert_array_equal(pop.historical_best_fitness, np.full(expected_shape[0], -np.inf))


def test_init_uses_default_encoding(dummy_objfunc):
    pop = Population(dummy_objfunc, np.eye(2))
    # DefaultEncoding should be a DefaultEncoding instance (imported)
    from metaheuristic_designer.encoding import DefaultEncoding

    assert isinstance(pop.encoding, DefaultEncoding)


def test_len(dummy_objfunc):
    pop = Population(dummy_objfunc, SMALL_GENOTYPE)
    assert len(pop) == 3


def test_iter(dummy_objfunc):
    pop = Population(dummy_objfunc, SMALL_GENOTYPE)
    rows = list(pop)
    np.testing.assert_array_equal(rows[0], SMALL_GENOTYPE[0])
    assert len(rows) == 3


# ---------------------------------------------------------------
# __copy__
# ---------------------------------------------------------------


@pytest.mark.parametrize(
    "genotype",
    [
        SMALL_GENOTYPE,
        LARGE_GENOTYPE,
        np.zeros((0, 2)),
    ],
)
def test_copy_creates_independent_object(genotype, dummy_objfunc):
    pop = Population(dummy_objfunc, genotype)
    # Set some non‑default values
    pop.fitness = np.array([1.0, 2.0, 3.0] if len(genotype) > 0 else [])
    pop.best = np.array([9.9])
    pop.best_fitness = 99.0
    pop.historical_best_matrix = np.ones_like(genotype)
    pop.historical_best_fitness = np.array([5.0] * len(genotype))

    pop2 = copy(pop)
    # Check equality of values
    np.testing.assert_array_equal(pop2.genotype_matrix, pop.genotype_matrix)
    np.testing.assert_array_equal(pop2.fitness, pop.fitness)
    np.testing.assert_array_equal(pop2.historical_best_matrix, pop.historical_best_matrix)
    assert pop2.best == pop.best
    assert pop2.best_fitness == pop.best_fitness

    # Modify the copy and ensure original is unaffected
    if len(genotype) > 0:
        pop2.genotype_matrix[0, 0] = -999.0
        assert pop.genotype_matrix[0, 0] != -999.0
        pop2.fitness[0] = -100.0
        assert pop.fitness[0] != -100.0
        pop2.historical_best_matrix[0, 0] = -555.0
        assert pop.historical_best_matrix[0, 0] != -555.0


# ---------------------------------------------------------------
# best_solution
# ---------------------------------------------------------------


@pytest.fixture
def pop_with_best(dummy_objfunc):
    pop = Population(dummy_objfunc, np.array([[0, 0], [1, 1]]))
    pop.fitness = np.array([10.0, 20.0])
    pop.best = np.array([1.0, 1.0])
    pop.best_fitness = 20.0
    return pop


def test_best_solution_max(pop_with_best):
    sol, fit = pop_with_best.best_solution()
    np.testing.assert_array_equal(sol, pop_with_best.best)
    assert fit == 20.0


def test_best_solution_min(dummy_objfunc_min):
    pop = Population(dummy_objfunc_min, np.array([[0, 0], [1, 1]]))
    pop.fitness = np.array([10.0, 20.0])
    pop.best = np.array([0.0, 0.0])
    pop.best_fitness = 10.0  # the best in min mode would be 10, but stored raw
    sol, _ = pop.best_solution(problem_space=True)
    _, fit = pop.best_solution(problem_space=False)
    assert fit == 10.0
    np.testing.assert_array_equal(sol, pop.best)


def test_best_solution_decoded(dummy_objfunc):
    pop = Population(dummy_objfunc, np.array([[2, 3]]))
    pop.fitness = np.array([42.0])
    pop.best = np.array([2.0, 3.0])
    pop.best_fitness = 42.0
    # DefaultEncoding returns the array unchanged when decoded
    sol, _ = pop.best_solution(problem_space=True)
    np.testing.assert_array_equal(sol, pop.best)


# ---------------------------------------------------------------
# update_genotype
# ---------------------------------------------------------------


@pytest.fixture
def pop_3(dummy_objfunc):
    pop = Population(dummy_objfunc, SMALL_GENOTYPE.copy())
    pop.fitness = np.array([-5.0, 0.0, 5.0])
    pop.fitness_calculated = np.ones(3, dtype=bool)
    return pop


def test_update_genotype_same_shape(pop_3):
    new_geno = np.array([[9, 9], [9, 9], [9, 9]])
    pop_3.update_genotype(new_geno)
    np.testing.assert_array_equal(pop_3.genotype_matrix, new_geno)
    # fitness_calculated should be all False because new_geno != old
    assert not np.any(pop_3.fitness_calculated)
    # fitness array stays the same
    np.testing.assert_array_equal(pop_3.fitness, [-5.0, 0.0, 5.0])


def test_update_genotype_same_values(pop_3):
    # When the new matrix is identical, fitness_calculated becomes True
    pop_3.update_genotype(SMALL_GENOTYPE.copy())
    assert np.all(pop_3.fitness_calculated)


def test_update_genotype_different_size(pop_3):
    new_geno = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    pop_3.update_genotype(new_geno)
    assert pop_3.pop_size == 4
    np.testing.assert_array_equal(pop_3.genotype_matrix, new_geno)
    np.testing.assert_array_equal(pop_3.fitness, np.full(4, -np.inf))
    np.testing.assert_array_equal(pop_3.fitness_calculated, np.zeros(4, dtype=bool))
    # historical best should be reset
    np.testing.assert_array_equal(pop_3.historical_best_matrix, new_geno)


def test_update_genotype_invalid_dim(pop_3):
    wrong_geno = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    with pytest.raises(ValueError):
        pop_3.update_genotype(wrong_geno)


def test_update_genotype_from_population(pop_3, dummy_objfunc):
    other_pop = Population(dummy_objfunc, np.array([[0, 0], [0, 0], [0, 0]]))
    pop_3.update_genotype(other_pop)
    np.testing.assert_array_equal(pop_3.genotype_matrix, other_pop.genotype_matrix)


# ---------------------------------------------------------------
# take_selection
# ---------------------------------------------------------------


@pytest.fixture
def populated_fit(dummy_objfunc):
    pop = Population(dummy_objfunc, np.arange(8).reshape(4, 2).astype(float))
    pop.fitness = np.array([3.0, 1.0, 4.0, 2.0])
    pop.historical_best_matrix = np.ones((4, 2))
    pop.historical_best_fitness = np.array([10.0, 20.0, 30.0, 40.0])
    pop.best = np.array([99.0, 99.0])
    pop.best_fitness = 99.0
    pop.fitness_calculated = np.array([True, False, True, False])
    return pop


@pytest.mark.parametrize(
    "sel_idx, expected_geno, expected_fit, expected_hist_best, expected_hist_best_fit",
    [
        # All rows in reverse order
        (
            [3, 2, 1, 0],
            np.array([[6, 7], [4, 5], [2, 3], [0, 1]]),
            np.array([2.0, 4.0, 1.0, 3.0]),
            np.ones((4, 2)),
            np.array([40.0, 30.0, 20.0, 10.0]),
        ),
        # Single row
        ([0], np.array([[0, 1]]), np.array([3.0]), np.ones((1, 2)), np.array([10.0])),
        # Empty selection
        ([], np.zeros((0, 2)), np.array([]), np.zeros((0, 2)), np.array([])),
        # Boolean mask
        ([True, False, True, False], np.array([[0, 1], [4, 5]]), np.array([3.0, 4.0]), np.ones((2, 2)), np.array([10.0, 30.0])),
    ],
)
def test_take_selection(populated_fit, sel_idx, expected_geno, expected_fit, expected_hist_best, expected_hist_best_fit):
    result = populated_fit.take_selection(np.array(sel_idx))
    assert result.pop_size == len(sel_idx)
    np.testing.assert_array_equal(result.genotype_matrix, expected_geno)
    np.testing.assert_array_equal(result.fitness, expected_fit)
    np.testing.assert_array_equal(result.historical_best_matrix, expected_hist_best)
    np.testing.assert_array_equal(result.historical_best_fitness, expected_hist_best_fit)
    # best is always a copy of the original best
    np.testing.assert_array_equal(result.best, np.array([99.0, 99.0]))
    assert result.best_fitness == 99.0


# ---------------------------------------------------------------
# apply_selection
# ---------------------------------------------------------------


def test_apply_selection(populated_fit, dummy_objfunc):
    # Create a donor population
    donor = Population(dummy_objfunc, np.array([[10, 11], [12, 13]]))
    donor.fitness = np.array([100.0, 200.0])
    donor.best = np.array([100.0, 200.0])
    donor.best_fitness = 300.0
    donor.historical_best_matrix = np.ones((2, 2))
    donor.historical_best_fitness = np.array([1.0, 2.0])
    donor.fitness_calculated = np.array([True, False])

    sel_idx = np.array([1, 3])  # replace second and fourth rows

    populated_fit.apply_selection(donor, sel_idx)

    # Check that rows are replaced
    np.testing.assert_array_equal(populated_fit.genotype_matrix[1], donor.genotype_matrix[0])
    np.testing.assert_array_equal(populated_fit.genotype_matrix[3], donor.genotype_matrix[1])
    assert populated_fit.fitness[1] == 100.0
    assert populated_fit.fitness[3] == 200.0
    assert populated_fit.historical_best_fitness[1] == 1.0
    assert populated_fit.historical_best_fitness[3] == 2.0
    # best should be updated because donor's best_fitness > original
    np.testing.assert_array_equal(populated_fit.best, np.array([100.0, 200.0]))
    assert populated_fit.best_fitness == 300.0


# ---------------------------------------------------------------
# take_slice / apply_slice
# ---------------------------------------------------------------


def test_take_slice(populated_fit):
    mask = np.array([1])  # keep only column 1
    sliced = populated_fit.take_slice(mask)
    assert sliced.vec_size == 1
    np.testing.assert_array_equal(sliced.genotype_matrix[:, 0], populated_fit.genotype_matrix[:, 1])
    # All other attributes are copied as row‑wise copies
    np.testing.assert_array_equal(sliced.fitness, populated_fit.fitness)


def test_apply_slice(populated_fit, dummy_objfunc):
    original = populated_fit.genotype_matrix.copy()
    mask = np.array([0])  # replace column 0
    donor = Population(dummy_objfunc, np.array([[100], [200], [300], [400]]))
    donor.best = np.array([100.0])
    donor.best_fitness = 999.0
    populated_fit.apply_slice(donor, mask)
    expected = original.copy()
    expected[:, 0] = donor.genotype_matrix[:, 0]
    np.testing.assert_array_equal(populated_fit.genotype_matrix, expected)
    # best updated because donor best_fitness > current best
    assert populated_fit.best_fitness == 999.0


# ---------------------------------------------------------------
# join_populations (static) and join (instance)
# ---------------------------------------------------------------


@pytest.fixture
def pop_a(dummy_objfunc):
    pop = Population(dummy_objfunc, np.array([[1, 1], [2, 2]]))
    pop.fitness = np.array([10.0, 20.0])
    pop.historical_best_matrix = np.ones((2, 2))
    pop.historical_best_fitness = np.array([30.0, 40.0])
    pop.best = np.array([99.0, 99.0])
    pop.best_fitness = 99.0
    pop.fitness_calculated = np.array([True, False])
    return pop


@pytest.fixture
def pop_b(dummy_objfunc):
    pop = Population(dummy_objfunc, np.array([[3, 3], [4, 4], [5, 5]]))
    pop.fitness = np.array([50.0, 60.0, 70.0])
    pop.historical_best_matrix = np.ones((3, 2)) * 2
    pop.historical_best_fitness = np.array([80.0, 90.0, 100.0])
    pop.best = np.array([100.0, 100.0])
    pop.best_fitness = 200.0
    pop.fitness_calculated = np.array([False, True, False])
    return pop


@pytest.mark.parametrize("method", ["static", "instance"])
def test_join_populations(pop_a, pop_b, method, dummy_objfunc):
    if method == "static":
        joined = Population.join_populations(pop_a, pop_b)
    else:
        # instance method modifies pop_a and returns it
        pop_a_copy = copy(pop_a)
        joined = pop_a_copy.join(pop_b)

    assert len(joined) == 5
    np.testing.assert_array_equal(joined.genotype_matrix[:2], pop_a.genotype_matrix)
    np.testing.assert_array_equal(joined.genotype_matrix[2:], pop_b.genotype_matrix)
    np.testing.assert_array_equal(joined.fitness, np.concatenate([pop_a.fitness, pop_b.fitness]))
    np.testing.assert_array_equal(joined.historical_best_fitness, np.concatenate([pop_a.historical_best_fitness, pop_b.historical_best_fitness]))
    # best should be the one with higher best_fitness (pop_b has 200 > 99)
    np.testing.assert_array_equal(joined.best, pop_b.best)
    assert joined.best_fitness == pop_b.best_fitness

    if method == "instance":
        # After instance join, pop_a_copy should be the same object as joined
        assert pop_a_copy is joined


def test_join_populations_best_tie(pop_a):
    other = copy(pop_a)
    other.fitness = np.array([5.0, 15.0])
    other.best_fitness = pop_a.best_fitness  # same
    joined = Population.join_populations(pop_a, other)
    # keep first's best when tied
    np.testing.assert_array_equal(joined.best, pop_a.best)


# ---------------------------------------------------------------
# sort_population
# ---------------------------------------------------------------


@pytest.fixture
def pop_unsorted(dummy_objfunc):
    pop = Population(dummy_objfunc, np.array([[1, 1], [2, 2], [3, 3], [4, 4]]))
    pop.fitness = np.array([-1.0, 3.0, 0.0, 2.0])
    pop.historical_best_fitness = np.array([10, 20, 30, 40])
    pop.historical_best_matrix = np.arange(8).reshape(4, 2).astype(float)
    pop.fitness_calculated = np.array([True, False, True, False])
    return pop


def test_sort_population(pop_unsorted):
    pop_unsorted.sort_population()
    # Expected: sorted by ascending fitness
    expected_order = np.argsort(
        [-1.0, 3.0, 0.0, 2.0]
    )  # [-1, 0, 2, 3] -> [-1,0,2,3] but argsort default ascending gives indices [0,2,3,1]? Let's compute: values: -1,3,0,2. Argsort ascending: [0(-1), 2(0), 3(2), 1(3)].
    expected_geno = np.array([[1, 1], [3, 3], [4, 4], [2, 2]])
    np.testing.assert_array_equal(pop_unsorted.genotype_matrix, expected_geno)
    np.testing.assert_array_equal(pop_unsorted.fitness, np.array([-1.0, 0.0, 2.0, 3.0]))
    np.testing.assert_array_equal(pop_unsorted.historical_best_fitness, np.array([10, 30, 40, 20]))
    np.testing.assert_array_equal(pop_unsorted.fitness_calculated, np.array([True, True, False, False]))


# ---------------------------------------------------------------
# update_best_from_parents
# ---------------------------------------------------------------


def test_update_best_from_parents(pop_a, pop_b):
    # pop_a best_fitness 99, pop_b best_fitness 200 -> update to pop_b's best
    pop_a.update_best_from_parents(pop_b)
    np.testing.assert_array_equal(pop_a.best, np.array([100.0, 100.0]))
    assert pop_a.best_fitness == 200.0


def test_update_best_from_parents_no_better(pop_a):
    worse = copy(pop_a)
    worse.best_fitness = 50.0
    original_best = pop_a.best.copy()
    pop_a.update_best_from_parents(worse)
    np.testing.assert_array_equal(pop_a.best, original_best)
    assert pop_a.best_fitness == 99.0


# ---------------------------------------------------------------
# step
# ---------------------------------------------------------------


def test_step_updates_best_and_calls_encoding_step(dummy_objfunc, simple_encoding):
    pop = Population(dummy_objfunc, np.array([[0, 0], [1, 1]]))
    pop.fitness = np.array([5.0, 10.0])
    pop.encoding = simple_encoding
    # simple_encoding.step returns the same genotype (identity)
    pop.step()
    # best should be the one with max fitness (index 1)
    np.testing.assert_array_equal(pop.best, np.array([1.0, 1.0]))
    assert pop.best_fitness == 10.0
    # genotype not changed because encoding.step is identity
    np.testing.assert_array_equal(pop.genotype_matrix, np.array([[0, 0], [1, 1]]))


def test_step_no_current_best(dummy_objfunc, simple_encoding):
    pop = Population(dummy_objfunc, np.array([[0, 0]]))
    pop.fitness = np.array([7.0])
    pop.best = None
    pop.best_fitness = None
    pop.encoding = simple_encoding
    pop.step()
    np.testing.assert_array_equal(pop.best, np.array([0.0, 0.0]))
    assert pop.best_fitness == 7.0


# ---------------------------------------------------------------
# repeat
# ---------------------------------------------------------------


@pytest.mark.parametrize(
    "amount, expected_rows",
    [
        (2, 6),
        (3, 9),
        (1, 3),  # amount=1 should still work (repeat once = original)
    ],
)
def test_repeat(amount, expected_rows, dummy_objfunc):
    pop = Population(dummy_objfunc, SMALL_GENOTYPE)
    pop.fitness = np.array([1.0, 2.0, 3.0])
    repeated = pop.repeat(amount)
    assert repeated.pop_size == expected_rows
    # Each original row repeated `amount` times sequentially
    for i in range(3):
        for j in range(amount):
            np.testing.assert_array_equal(repeated.genotype_matrix[i * amount + j], SMALL_GENOTYPE[i])


# ---------------------------------------------------------------
# calculate_fitness
# ---------------------------------------------------------------


def test_calculate_fitness_updates_historical_and_best(dummy_objfunc):
    """Simulate the objective function returning new fitness values."""
    pop = Population(dummy_objfunc, np.array([[0, 0], [1, 1]]))
    pop.fitness = np.array([2.0, 5.0])
    pop.best = np.array([1.0, 1.0])
    pop.best_fitness = 5.0
    pop.historical_best_matrix = np.array([[0, 0], [1, 1]])
    pop.historical_best_fitness = np.array([2.0, 5.0])

    # Mock to return higher fitness for first individual, lower for second
    dummy_objfunc._fitness_return = np.array([10.0, 1.0])
    pop.calculate_fitness(parallel=False)

    np.testing.assert_array_equal(pop.fitness, [10.0, 1.0])
    # historical best should be updated for improved individuals (first only)
    np.testing.assert_array_equal(pop.historical_best_matrix[0], [0, 0])  # still
    assert pop.historical_best_fitness[0] == 10.0
    np.testing.assert_array_equal(pop.historical_best_matrix[1], [1, 1])  # unchanged
    assert pop.historical_best_fitness[1] == 5.0
    # best updated because 10.0 > 5.0
    np.testing.assert_array_equal(pop.best, np.array([0.0, 0.0]))
    assert pop.best_fitness == 10.0


# ---------------------------------------------------------------
# repair_solutions
# ---------------------------------------------------------------


def test_repair_solutions(dummy_objfunc):
    dummy_objfunc._repair_return = lambda x: x * 2
    pop = Population(dummy_objfunc, np.array([[1, 2], [3, 4]]))
    pop.repair_solutions()
    np.testing.assert_array_equal(pop.genotype_matrix, [[2, 4], [6, 8]])


# ---------------------------------------------------------------
# decode / encode / decode_params (delegation to encoding)
# ---------------------------------------------------------------


def test_decode_uses_default_encoding(dummy_objfunc):
    pop = Population(dummy_objfunc, np.array([[0, 0]]))
    decoded = pop.decode()
    np.testing.assert_array_equal(decoded, pop.genotype_matrix)


def test_encode_uses_default_encoding(dummy_objfunc):
    pop = Population(dummy_objfunc, np.array([[0, 0]]))
    encoded = pop.encode()
    np.testing.assert_array_equal(encoded, pop.genotype_matrix)


# ---------------------------------------------------------------
# get_state
# ---------------------------------------------------------------


def test_get_state(dummy_objfunc):
    pop = Population(dummy_objfunc, SMALL_GENOTYPE)
    pop.fitness = np.array([1.0, 2.0, 3.0])
    pop.best = np.array([4.0, 5.0])
    pop.best_fitness = 6.0
    state = pop.get_state()
    assert "genotype_matrix" in state
    assert "fitness" in state
    assert "best" in state
    assert "encoding" in state
    assert state["encoding"] == "DefaultEncoding"
