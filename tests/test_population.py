import pytest
import numpy as np
from copy import copy
import sys
from unittest.mock import Mock

# Mock the dependencies since we're testing in isolation
sys.modules['.utils'] = Mock()
sys.modules['.objective_function'] = Mock()
sys.modules['.encoding'] = Mock()

from metaheuristic_designer import RAND_GEN, ObjectiveFunc, Encoding, DefaultEncoding, Population
from metaheuristic_designer.encodings import ParameterExtendingEncoding


# Fixtures for common test objects
@pytest.fixture
def mock_objfunc():
    objfunc = Mock(spec=ObjectiveFunc)
    objfunc.name = "Mock function"
    objfunc.mode = "min"
    objfunc.fitness.return_value = np.array([1.0, 2.0, 3.0])
    objfunc.repair_solution.side_effect = lambda x: x  # Identity repair
    return objfunc


@pytest.fixture
def sample_genotype():
    return np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


@pytest.fixture
def default_encoding():
    return DefaultEncoding()


# Test basic initialization and properties
@pytest.mark.parametrize("pop_size, vec_size", [
    (3, 3),
    (5, 10),
    (1, 1),
])
def test_population_initialization(mock_objfunc, pop_size, vec_size):
    genotype = np.random.rand(pop_size, vec_size)
    pop = Population(mock_objfunc, genotype)
    
    assert pop.objfunc == mock_objfunc
    assert pop.genotype_matrix.shape == (pop_size, vec_size)
    assert pop.pop_size == pop_size
    assert pop.vec_size == vec_size
    assert len(pop.fitness) == pop_size
    assert len(pop.fitness_calculated) == pop_size
    assert np.all(pop.fitness == -np.inf)
    assert np.all(pop.fitness_calculated == 0)
    assert pop.best is None
    assert pop.best_fitness is None


def test_population_length(mock_objfunc, sample_genotype):
    pop = Population(mock_objfunc, sample_genotype)
    assert len(pop) == 3


def test_population_iteration(mock_objfunc, sample_genotype):
    pop = Population(mock_objfunc, sample_genotype)
    genotypes = list(pop)
    assert len(genotypes) == 3
    assert np.array_equal(genotypes[0], sample_genotype[0])
    assert np.array_equal(genotypes[1], sample_genotype[1])
    assert np.array_equal(genotypes[2], sample_genotype[2])


def test_population_copy(mock_objfunc, sample_genotype):
    pop = Population(mock_objfunc, sample_genotype)
    pop.fitness = np.array([1.0, 2.0, 3.0])
    pop.best = sample_genotype[0]
    pop.best_fitness = 1.0
    
    copied_pop = copy(pop)
    
    # Check it's a different object
    assert copied_pop is not pop
    # Check attributes are equal
    assert np.array_equal(copied_pop.genotype_matrix, pop.genotype_matrix)
    assert np.array_equal(copied_pop.fitness, pop.fitness)
    assert np.array_equal(copied_pop.best, pop.best)
    assert copied_pop.best_fitness == pop.best_fitness


def test_population_repr(mock_objfunc, sample_genotype):
    pop = Population(mock_objfunc, sample_genotype)
    pop.fitness = np.array([1.0, 2.0, 3.0])
    pop.best = sample_genotype[0]
    pop.best_fitness = 1.0
    
    repr_str = repr(pop)
    
    assert "Population" in repr_str
    assert "objfunc" in repr_str
    assert "genotype_matrix" in repr_str
    assert "fitness" in repr_str
    assert "best" in repr_str
    assert "best_fitness" in repr_str


@pytest.mark.parametrize("mode,expected_fitness", [
    ("min", -3.0),  # In min mode, best_fitness should be multiplied by -1
    ("max", 3.0),
])
def test_best_solution(mock_objfunc, sample_genotype, mode, expected_fitness):
    mock_objfunc.mode = mode
    pop = Population(mock_objfunc, sample_genotype)
    pop.best = sample_genotype[2]  # Set best to last individual
    pop.best_fitness = 3.0
    
    best_sol, best_fit = pop.best_solution(decoded=False)
    
    assert np.array_equal(best_sol, sample_genotype[2])
    assert best_fit == expected_fitness


@pytest.mark.parametrize("new_size,same_size", [
    (3, True),  # Same size update
    (5, False),  # Different size update
])
def test_update_genotype_matrix(mock_objfunc, sample_genotype, new_size, same_size):
    pop = Population(mock_objfunc, sample_genotype)
    new_genotype = np.random.rand(new_size, 3)
    
    result = pop.update_genotype_matrix(new_genotype)
    
    assert result is pop
    assert np.array_equal(pop.genotype_matrix, new_genotype)
    assert pop.pop_size == new_size
    
    if same_size:
        # Fitness calculated should be based on element-wise comparison
        assert len(pop.fitness_calculated) == new_size
    else:
        # Fitness arrays should be reset
        assert np.all(pop.fitness == -np.inf)
        assert np.all(pop.fitness_calculated == 0)


@pytest.mark.parametrize("selection_idx", [
    [0, 2],           # List of indices
    np.array([0, 2]), # Array of indices  
    [True, False, True],  # Boolean mask
])
def test_take_selection(mock_objfunc, sample_genotype, selection_idx):
    pop = Population(mock_objfunc, sample_genotype)
    pop.fitness = np.array([1.0, 2.0, 3.0])
    pop.best = sample_genotype[2]
    pop.best_fitness = 3.0
    
    selected = pop.take_selection(selection_idx)
    
    expected_indices = np.array(selection_idx) if isinstance(selection_idx, list) and all(isinstance(x, bool) for x in selection_idx) else selection_idx
    expected_genotype = pop.genotype_matrix[expected_indices]
    
    assert np.array_equal(selected.genotype_matrix, expected_genotype)
    assert np.array_equal(selected.fitness, pop.fitness[expected_indices])
    assert selected.best_fitness == pop.best_fitness


@pytest.mark.parametrize("selection_idx", [
    [0, 2],
    [True, False, True],
])
def test_apply_selection(mock_objfunc, sample_genotype, selection_idx):
    pop = Population(mock_objfunc, sample_genotype)
    selected_pop = Population(mock_objfunc, np.array([[10, 11, 12], [13, 14, 15]]))
    selected_pop.best = np.array([13, 14, 15])
    selected_pop.best_fitness = 5.0
    
    result = pop.apply_selection(selected_pop, selection_idx)
    
    assert result is pop
    expected_indices = selection_idx if isinstance(selection_idx, list) and all(isinstance(x, bool) for x in selection_idx) else selection_idx
    
    # Check that selected indices were updated
    assert np.array_equal(pop.genotype_matrix[expected_indices], selected_pop.genotype_matrix)
    assert np.array_equal(pop.fitness[expected_indices], selected_pop.fitness)
    
    # Best should be updated since selected_pop has better fitness
    assert np.array_equal(pop.best, selected_pop.best)
    assert pop.best_fitness == selected_pop.best_fitness


@pytest.mark.parametrize("mask", [
    [0, 2],           # List of indices
    np.array([0, 2]), # Array of indices
    [True, False, True],  # Boolean mask
])
def test_take_slice(mock_objfunc, sample_genotype, mask):
    pop = Population(mock_objfunc, sample_genotype)
    
    sliced = pop.take_slice(mask)
    
    expected_mask = mask if isinstance(mask, list) and all(isinstance(x, bool) for x in mask) else mask
    expected_genotype = pop.genotype_matrix[:, expected_mask]
    
    assert np.array_equal(sliced.genotype_matrix, expected_genotype)
    if isinstance(mask[0], bool):
        assert sliced.vec_size == sum(expected_mask)
    else:
        assert sliced.vec_size == len(expected_mask)


def test_join_populations_static(mock_objfunc, sample_genotype):
    pop1 = Population(mock_objfunc, sample_genotype)
    pop1.fitness = np.array([1.0, 2.0, 3.0])
    pop1.best = sample_genotype[2]
    pop1.best_fitness = 3.0

    # Make pop2 have the better best fitness
    pop2 = Population(mock_objfunc, np.array([[10, 11, 12]]))
    pop2.fitness = np.array([4.0])
    pop2.best = np.array([10, 11, 12])
    pop2.best_fitness = 4.0
    
    joined = Population.join_populations(pop1, pop2)
    
    expected_genotype = np.vstack([sample_genotype, [[10, 11, 12]]])
    expected_fitness = np.array([1.0, 2.0, 3.0, 4.0])
    
    assert np.array_equal(joined.genotype_matrix, expected_genotype)
    assert np.array_equal(joined.fitness, expected_fitness)
    assert joined.pop_size == 4
    # Best should come from pop2 since it has better fitness
    assert np.array_equal(joined.best, pop2.best)
    assert joined.best_fitness == pop2.best_fitness


def test_join_method(mock_objfunc, sample_genotype):
    pop1 = Population(mock_objfunc, sample_genotype)
    pop2 = Population(mock_objfunc, np.array([[10, 11, 12]]))
    
    result = pop1.join(pop2)
    
    assert result is pop1
    assert pop1.pop_size == 4
    assert np.array_equal(pop1.genotype_matrix[-1], [10, 11, 12])


def test_sort_population(mock_objfunc, sample_genotype):
    pop = Population(mock_objfunc, sample_genotype)
    pop.fitness = np.array([3.0, 1.0, 2.0])  # Unsorted fitness
    
    result = pop.sort_population()
    
    assert result is pop
    # After sorting, genotype should be ordered by fitness
    expected_order = np.argsort([3.0, 1.0, 2.0])
    expected_genotype = sample_genotype[expected_order]
    assert np.array_equal(pop.genotype_matrix, expected_genotype)
    assert np.array_equal(pop.fitness, np.array([1.0, 2.0, 3.0]))

def test_best_solution_decoded(mock_objfunc, sample_genotype, default_encoding):
    pop = Population(mock_objfunc, sample_genotype, encoding=default_encoding)
    pop.best = sample_genotype[1]  # Set best to second individual
    pop.best_fitness = 2.0
    
    # Mock the decode method
    decoded_value = np.array([[100, 200, 300]])
    default_encoding.decode = Mock(return_value=decoded_value)
    
    best_sol, best_fit = pop.best_solution(decoded=True)
    
    assert np.array_equal(best_sol, decoded_value[0])
    assert best_fit == -2.0  # Since mode is "min" in mock_objfunc, but we're checking the decoded path


@pytest.mark.parametrize("encoding_type,expected_return", [
    (ParameterExtendingEncoding, np.array([1, 2, 3])),  # Should return decoded params
    (DefaultEncoding, None),  # Should return None
])
def test_decode_params(mock_objfunc, sample_genotype, encoding_type, expected_return):
    encoding = Mock(spec=encoding_type)
    if encoding_type == ParameterExtendingEncoding:
        encoding.decode_params.return_value = expected_return
    else:
        # For non-ExtendedEncoding, the method shouldn't exist in the same way
        pass
    
    pop = Population(mock_objfunc, sample_genotype, encoding=encoding)
    
    result = pop.decode_params()
    
    if encoding_type == ParameterExtendingEncoding:
        assert np.array_equal(result, expected_return)
        encoding.decode_params.assert_called_once_with(sample_genotype)
    else:
        assert result is None


def test_update_best_from_parents(mock_objfunc, sample_genotype):
    pop = Population(mock_objfunc, sample_genotype)
    parents = Population(mock_objfunc, sample_genotype)
    parents.best = np.array([10, 11, 12])
    parents.best_fitness = 5.0  # Better than pop's current best (None)
    
    result = pop.update_best_from_parents(parents)
    
    assert result is pop
    assert np.array_equal(pop.best, parents.best)
    assert pop.best_fitness == parents.best_fitness


def test_update(mock_objfunc, sample_genotype):
    pop = Population(mock_objfunc, sample_genotype)
    pop.fitness = np.array([1.0, 3.0, 2.0])  # Best is at index 1
    
    # Mock the encoding update to return the same matrix
    pop.encoding.update = Mock(return_value=sample_genotype)
    
    result = pop.update()
    
    assert result is pop
    # Best should be set to individual at index 1
    assert np.array_equal(pop.best, sample_genotype[1])
    assert pop.best_fitness == 3.0


@pytest.mark.parametrize("amount", [1, 2, 3])
def test_repeat(mock_objfunc, sample_genotype, amount):
    pop = Population(mock_objfunc, sample_genotype)
    
    repeated = pop.repeat(amount)
    
    expected_genotype = np.tile(sample_genotype, (amount, 1))
    assert np.array_equal(repeated.genotype_matrix, expected_genotype)
    assert repeated.pop_size == 3 * amount


def test_calculate_fitness(mock_objfunc, sample_genotype):
    pop = Population(mock_objfunc, sample_genotype)
    
    result = pop.calculate_fitness()
    
    assert result is pop
    # Fitness should be set to what objfunc.fitness returns
    assert np.array_equal(pop.fitness, np.array([1.0, 2.0, 3.0]))
    # Historical best should be updated
    assert np.array_equal(pop.historical_best_fitness, np.array([1.0, 2.0, 3.0]))
    # Best individual should be set
    assert np.array_equal(pop.best, sample_genotype[2])  # Index 2 has fitness 3.0
    assert pop.best_fitness == 3.0


def test_repair_solutions(mock_objfunc, sample_genotype):
    pop = Population(mock_objfunc, sample_genotype)
    
    result = pop.repair_solutions()
    
    assert result is pop
    # Since repair_solution is identity mock, genotype should be unchanged
    assert np.array_equal(pop.genotype_matrix, sample_genotype)
    # Verify repair_solution was called for each individual
    assert mock_objfunc.repair_solution.call_count == 3


def test_decode(mock_objfunc, sample_genotype, default_encoding):
    pop = Population(mock_objfunc, sample_genotype, encoding=default_encoding)
    
    # Mock the decode method
    decoded_value = np.array([[1, 2], [3, 4], [5, 6]])
    default_encoding.decode = Mock(return_value=decoded_value)
    
    result = pop.decode()
    
    assert np.array_equal(result, decoded_value)
    default_encoding.decode.assert_called_once_with(sample_genotype)


def test_get_state(mock_objfunc, sample_genotype):
    pop = Population(mock_objfunc, sample_genotype)
    pop.fitness = np.array([1.0, 2.0, 3.0])
    pop.best = sample_genotype[0]
    pop.best_fitness = 1.0
    
    state = pop.get_state()
    
    expected_state = {
        "genotype_matrix": sample_genotype,
        "fitness": np.array([1.0, 2.0, 3.0]),
        "historical_best_matrix": sample_genotype,
        "historical_best_fitness": pop.historical_best_fitness,
        "best": sample_genotype[0],
        "best_fitness": 1.0,
        "encoding": "DefaultEncoding",
    }
    
    for key in expected_state:
        if isinstance(expected_state[key], np.ndarray):
            assert np.array_equal(state[key], expected_state[key])
        else:
            assert state[key] == expected_state[key]


# Test error cases
def test_update_genotype_matrix_invalid_size(mock_objfunc, sample_genotype):
    pop = Population(mock_objfunc, sample_genotype)
    invalid_genotype = np.random.rand(3, 5)  # Different vector size
    
    with pytest.raises(ValueError, match="Individual vector size should not change"):
        pop.update_genotype_matrix(invalid_genotype)