"""
Unit tests for utils.py.

Contracts verified:
- NumpyEncoder serialises numpy scalars, arrays, and enums to JSON.
- check_random_state returns an np.random.Generator from None, int, or an existing generator.
- check_random_state raises ValueError for invalid seeds.
- per_individual applies a function row-wise.
- per_individual_list applies a function element-wise.
"""

import json
from enum import Enum

import numpy as np
import pytest

from metaheuristic_designer.utils import (
    NumpyEncoder,
    check_random_state,
    per_individual,
    per_individual_list,
)


class _Color(Enum):
    RED = 1
    BLUE = 2


# ---------------------------------------------------------------------------
# NumpyEncoder
# ---------------------------------------------------------------------------

def test_numpy_encoder_int32():
    data = np.int32(7)
    out = json.loads(json.dumps(data, cls=NumpyEncoder))
    assert out == 7
    assert isinstance(out, int)


def test_numpy_encoder_float64():
    data = np.float64(3.14)
    out = json.loads(json.dumps(data, cls=NumpyEncoder))
    assert out == pytest.approx(3.14)


def test_numpy_encoder_ndarray():
    arr = np.array([1, 2, 3])
    out = json.loads(json.dumps(arr, cls=NumpyEncoder))
    assert out == [1, 2, 3]


def test_numpy_encoder_enum():
    encoded = json.dumps(_Color.RED, cls=NumpyEncoder)
    assert "RED" in encoded


def test_numpy_encoder_unknown_type_raises():
    with pytest.raises(TypeError):
        json.dumps(object(), cls=NumpyEncoder)


# ---------------------------------------------------------------------------
# check_random_state
# ---------------------------------------------------------------------------

def test_check_random_state_with_none_returns_generator():
    rng = check_random_state(None)
    assert isinstance(rng, np.random.Generator)


def test_check_random_state_with_numpy_random_returns_generator():
    rng = check_random_state(np.random)
    assert isinstance(rng, np.random.Generator)


def test_check_random_state_with_int_is_deterministic():
    r1 = check_random_state(42).random()
    r2 = check_random_state(42).random()
    assert r1 == r2


def test_check_random_state_with_generator_passthrough():
    gen = np.random.default_rng(99)
    result = check_random_state(gen)
    assert result is gen


def test_check_random_state_invalid_seed_raises():
    with pytest.raises(ValueError):
        check_random_state("not_valid")


def test_check_random_state_float_raises():
    with pytest.raises(ValueError):
        check_random_state(3.14)


# ---------------------------------------------------------------------------
# per_individual
# ---------------------------------------------------------------------------

def test_per_individual_applies_row_wise():
    @per_individual
    def row_sum(row):
        return np.sum(row)

    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    result = row_sum(matrix)
    np.testing.assert_array_equal(result, [6, 15])


def test_per_individual_passes_kwargs():
    @per_individual
    def scale(row, factor=1):
        return row * factor

    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = scale(matrix, factor=3)
    np.testing.assert_array_equal(result, [[3.0, 6.0], [9.0, 12.0]])


# ---------------------------------------------------------------------------
# per_individual_list
# ---------------------------------------------------------------------------

def test_per_individual_list_applies_element_wise():
    @per_individual_list
    def double(x):
        return x * 2

    result = double([1, 2, 3])
    assert result == [2, 4, 6]


def test_per_individual_list_passes_kwargs():
    @per_individual_list
    def add(x, offset=0):
        return x + offset

    result = add([10, 20, 30], offset=5)
    assert result == [15, 25, 35]
