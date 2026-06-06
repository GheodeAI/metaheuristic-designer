import enum
import json
import pytest
import numpy as np
from metaheuristic_designer.utils import (
    NumpyEncoder,
    check_random_state,
    per_individual,
    per_individual_list,
)

def test_random_state_fail():
    with pytest.raises(ValueError):
        check_random_state("not valid")

def test_per_individual():
    def add_one(x):
        return np.zeros_like(x)
    
    vec_add_one = per_individual(add_one)

    matrix = np.eye(5)
    result = vec_add_one(matrix)

    assert isinstance(result, np.ndarray)
    assert result.ndim==2
    np.testing.assert_equal(result, np.zeros((5,5)))
    
def test_per_individual_list():
    def add_one(x):
        return np.zeros_like(x)
    
    vec_add_one = per_individual_list(add_one)

    matrix = np.eye(5)
    result = vec_add_one(matrix)

    assert isinstance(result, list)
    for row in result:
        np.testing.assert_equal(row, np.zeros(5))

def test_numpy_json_encoder_ints():
    data = {"np.int": np.ones(1, dtype=int)[0]}
    encoded = json.dumps(data, cls=NumpyEncoder)

    assert encoded == "{\"np.int\": 1}"

def test_numpy_json_encoder_floats():
    data = {"np.float": np.pi}
    encoded = json.dumps(data, cls=NumpyEncoder)

    assert encoded == "{\"np.float\": " + str(np.pi) + "}"

def test_numpy_json_encoder_enum():
    enum_example = enum.Enum('E', 'A')

    data = {"Enum": enum_example.A}
    encoded = json.dumps(data, cls=NumpyEncoder)

    assert encoded == "{\"Enum\": \"E.A\"}"

def test_numpy_json_encoder_array():
    data = {"np.array": np.eye(2, dtype=int)}
    encoded = json.dumps(data, cls=NumpyEncoder)

    assert encoded == "{\"np.array\": [[1, 0], [0, 1]]}"

def test_numpy_json_encoder_fallback():
    data = {"other": {"1": [3,4,5]}}
    encoded = json.dumps(data, cls=NumpyEncoder)
