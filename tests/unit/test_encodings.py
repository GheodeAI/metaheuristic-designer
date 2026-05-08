"""
Unit tests for encoding classes.

Covers:
- SigmoidEncoding: encode, decode (as_probability and as_binary modes)
- CompositeEncoding: encode, decode, encode_func, decode_func, extract_solution, extract_params
- TypeCastEncoding: encode, decode
"""

import numpy as np
import pytest

from metaheuristic_designer.encodings import TypeCastEncoding
from metaheuristic_designer.encodings.sigmoid_encoding import SigmoidEncoding
from metaheuristic_designer.encodings.composite_encoding import CompositeEncoding
from metaheuristic_designer.encodings.parameter_extending_encoding import ParameterExtendingEncoding
from metaheuristic_designer.encoding import DefaultEncoding


# ---------------------------------------------------------------------------
# SigmoidEncoding
# ---------------------------------------------------------------------------

def test_sigmoid_encoding_decode_as_probability_returns_values_in_01():
    enc = SigmoidEncoding(as_probability=True)
    pop = np.array([[0.0, 1.0, -1.0, 5.0, -5.0]])
    decoded = enc.decode(pop)
    assert np.all(decoded >= 0) and np.all(decoded <= 1)


def test_sigmoid_encoding_decode_as_probability_midpoint():
    enc = SigmoidEncoding(as_probability=True)
    pop = np.array([[0.0]])
    decoded = enc.decode(pop)
    assert decoded[0, 0] == pytest.approx(0.5, abs=1e-6)


def test_sigmoid_encoding_decode_as_binary_threshold_default():
    enc = SigmoidEncoding(as_probability=False, threshold=0.5)
    # logit(0.9) >> 0, should decode to 1
    # logit(0.1) << 0, should decode to 0
    pop = np.array([[2.0, -2.0]])  # sigmoid(2)>0.5, sigmoid(-2)<0.5
    decoded = enc.decode(pop)
    assert decoded[0, 0] == 1
    assert decoded[0, 1] == 0


def test_sigmoid_encoding_decode_as_binary_returns_int():
    enc = SigmoidEncoding(as_probability=False)
    pop = np.array([[1.0, -1.0]])
    decoded = enc.decode(pop)
    assert decoded.dtype in (np.int32, np.int64, int) or np.issubdtype(decoded.dtype, np.integer)


def test_sigmoid_encoding_encode_inverts_decode():
    enc = SigmoidEncoding(as_probability=True)
    # Encode takes values in (0,1)
    solutions = np.array([[0.1, 0.5, 0.9]])
    encoded = enc.encode(solutions)
    decoded = enc.decode(encoded)
    np.testing.assert_allclose(decoded, solutions, atol=1e-6)


def test_sigmoid_encoding_encode_rejects_values_outside_01():
    enc = SigmoidEncoding(as_probability=True)
    bad = np.array([[1.5, -0.1]])
    with pytest.raises(AssertionError):
        enc.encode(bad)


def test_sigmoid_encoding_threshold_assertion():
    with pytest.raises(AssertionError):
        SigmoidEncoding(as_probability=False, threshold=0.0)
    with pytest.raises(AssertionError):
        SigmoidEncoding(as_probability=False, threshold=1.0)


def test_sigmoid_encoding_large_positive_input_decodes_to_one():
    enc = SigmoidEncoding(as_probability=False, threshold=0.5)
    pop = np.array([[100.0]])
    decoded = enc.decode(pop)
    assert decoded[0, 0] == 1


def test_sigmoid_encoding_large_negative_input_decodes_to_zero():
    enc = SigmoidEncoding(as_probability=False, threshold=0.5)
    pop = np.array([[-100.0]])
    decoded = enc.decode(pop)
    assert decoded[0, 0] == 0


# ---------------------------------------------------------------------------
# TypeCastEncoding
# ---------------------------------------------------------------------------

def test_type_cast_encoding_float_to_int():
    enc = TypeCastEncoding(float, int)
    pop = np.array([[1.7, 2.3, -0.9]])
    decoded = enc.decode(pop)
    assert np.issubdtype(decoded.dtype, np.integer)


def test_type_cast_encoding_int_to_bool():
    enc = TypeCastEncoding(int, bool)
    pop = np.array([[1, 0, 1, 0]])
    decoded = enc.decode(pop)
    assert decoded.dtype == bool


def test_type_cast_encoding_roundtrip_float_to_int():
    enc = TypeCastEncoding(float, int)
    solutions = np.array([[3.0, 5.0, 7.0]])
    encoded = enc.encode(solutions)
    decoded = enc.decode(encoded)
    np.testing.assert_array_equal(decoded, solutions.astype(int))


# ---------------------------------------------------------------------------
# CompositeEncoding
# ---------------------------------------------------------------------------

def _make_simple_composite():
    """Two type cast encodings chained: float->int (outer), int->bool (inner — not ParameterExtending)."""
    enc_outer = TypeCastEncoding(float, int)
    enc_inner = TypeCastEncoding(int, bool)
    return CompositeEncoding([enc_outer, enc_inner])


def test_composite_encoding_construction():
    enc = _make_simple_composite()
    assert enc is not None


def test_composite_encoding_decode_applies_outer_then_inner():
    """decode() applies encodings in order (outer first, then inner in CompositeEncoding)."""
    enc = _make_simple_composite()
    pop = np.array([[1.9, 0.1, 3.5]])
    decoded = enc.decode(pop)
    # Should be a bool array (after both casts)
    assert decoded.dtype == bool or np.issubdtype(decoded.dtype, np.bool_)


@pytest.mark.xfail(
    reason="BUG: CompositeEncoding.decode_func calls encoding.decode_func() on non-ParameterExtendingEncoding "
           "members, but the base Encoding class has no decode_func method (AttributeError). See ERRORES.md."
)
def test_composite_encoding_decode_func_applies_in_reverse():
    """decode_func() applies encodings in reversed order."""
    enc = _make_simple_composite()
    pop = np.array([[1.9, 0.0, 2.1]])
    decoded = enc.decode_func(pop)
    assert decoded is not None


@pytest.mark.xfail(
    reason="BUG: CompositeEncoding.encode_func calls encoding.encode_func() on non-ParameterExtendingEncoding "
           "members, but the base Encoding class has no encode_func method (AttributeError). See ERRORES.md."
)
def test_composite_encoding_encode_func_applies_in_reverse():
    enc = _make_simple_composite()
    solutions = np.array([[1.0, 0.0, 1.0]])
    encoded = enc.encode_func(solutions)
    assert encoded is not None


def test_composite_encoding_encode_applies_in_reverse():
    enc = _make_simple_composite()
    solutions = np.array([[1.0, 0.0, 1.0]])
    encoded = enc.encode(solutions)
    assert encoded is not None
    assert encoded.shape == solutions.shape


def test_composite_encoding_with_parameter_extending():
    """CompositeEncoding with a ParameterExtendingEncoding member propagates param_sizes."""
    from metaheuristic_designer.encodings.special.self_adapting_ES_encoding import SelfAdaptingESEncoding

    # Self-adapting ES encoding is a ParameterExtendingEncoding
    inner = SelfAdaptingESEncoding(dimension=4)
    composite = CompositeEncoding([inner])

    assert composite.dimension == 4
    assert len(composite.param_sizes) > 0


def test_composite_encoding_extract_solution_and_params():
    """extract_solution and extract_params work on a plain CompositeEncoding."""
    from metaheuristic_designer.encodings.special.self_adapting_ES_encoding import SelfAdaptingESEncoding

    dim = 4
    inner = SelfAdaptingESEncoding(dimension=dim)
    composite = CompositeEncoding([inner])

    # An extended population has dim + nparams columns
    n_rows = 3
    total_cols = dim + composite.nparams
    pop_matrix = np.ones((n_rows, total_cols))

    solution_part = composite.extract_solution(pop_matrix)
    params_part = composite.extract_params(pop_matrix)

    assert solution_part.shape == (n_rows, dim)
    assert params_part.shape[0] == n_rows


def test_composite_encoding_without_param_extending_has_none_dimension():
    """Without any ParameterExtendingEncoding, dimension stays None."""
    enc = _make_simple_composite()
    assert enc.dimension is None


# ---------------------------------------------------------------------------
# DefaultEncoding
# ---------------------------------------------------------------------------

def test_default_encoding_decode_returns_input():
    enc = DefaultEncoding()
    pop = np.array([[1.0, 2.0, 3.0]])
    decoded = enc.decode(pop)
    np.testing.assert_array_equal(decoded, pop)


def test_default_encoding_encode_returns_input():
    enc = DefaultEncoding()
    solutions = np.array([[4.0, 5.0, 6.0]])
    encoded = enc.encode(solutions)
    np.testing.assert_array_equal(encoded, solutions)
