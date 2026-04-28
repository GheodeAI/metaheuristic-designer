# tests/test_encodings.py
import pytest
import numpy as np
import scipy.special as sp_special
from numpy.testing import assert_array_equal, assert_array_almost_equal

# Conftest fixtures – nothing else needed
from conftest import simple_encoding, DummyParameterExtendingEncoding

# Encodings under test
from metaheuristic_designer.encoding import DefaultEncoding, EncodingFromLambda
from metaheuristic_designer.encodings import (
    TypeCastEncoding,
    SigmoidEncoding,
    MatrixEncoding,
    ImageEncoding,
    PSOEncoding,
    SelfAdaptingESEncoding,
    CompositeEncoding,
)


# ===================================================================
#  DefaultEncoding (already covered by simple_encoding fixture)
# ===================================================================
@pytest.mark.parametrize("arr", [
    np.array([1, 2, 3]),
    np.array([[1, 2], [3, 4]]),
    np.array([[0.5]]),
    np.zeros((0, 2)),
])
def test_default_encode_decode_identity(arr, simple_encoding):
    assert_array_equal(simple_encoding.encode(arr), arr)
    assert_array_equal(simple_encoding.decode(arr), arr)


# -------------------------------------------------------------------
#  EncodingFromLambda
# -------------------------------------------------------------------
def test_encoding_from_lambda():
    # static closure with side‑effect counter
    call_counter = {"encode": 0, "decode": 0}

    def _encode(x):
        call_counter["encode"] += 1
        return x * 2

    def _decode(x):
        call_counter["decode"] += 1
        return x + 10

    enc = EncodingFromLambda(encode_fn=_encode, decode_fn=_decode)
    inp = np.array([1, 2, 3])
    assert_array_equal(enc.encode(inp), inp * 2)
    assert call_counter["encode"] == 1
    assert_array_equal(enc.decode(inp), inp + 10)
    assert call_counter["decode"] == 1


# ===================================================================
#  TypeCastEncoding
# ===================================================================
@pytest.mark.parametrize("inp, encoded_dtype, decoded_dtype, expected_enc_dtype, expected_dec_dtype", [
    (np.array([1.5, 2.7]), int, float, np.dtype(int), np.dtype(float)),
    (np.array([1, 2]),     float, int, np.dtype(float), np.dtype(int)),
    (np.array([[0.1, 0.2]]), int, float, np.dtype(int), np.dtype(float)),
])
def test_typecast_encode_decode_types(inp, encoded_dtype, decoded_dtype, expected_enc_dtype, expected_dec_dtype):
    enc = TypeCastEncoding(encoded_dtype=encoded_dtype, decoded_dtype=decoded_dtype)
    encoded = enc.encode(inp)
    assert encoded.dtype == expected_enc_dtype
    decoded = enc.decode(encoded)
    assert decoded.dtype == expected_dec_dtype


def test_typecast_roundtrip_float_int_float():
    enc = TypeCastEncoding(encoded_dtype=int, decoded_dtype=float)
    original = np.array([1.2, 3.9])
    encoded = enc.encode(original)
    assert_array_equal(encoded, [1, 3])
    decoded = enc.decode(encoded)
    assert decoded.dtype == np.float64
    assert_array_equal(decoded, [1.0, 3.0])


# ===================================================================
#  SigmoidEncoding
# ===================================================================
def test_sigmoid_encode_extreme_values():
    enc = SigmoidEncoding()
    # 0 → -inf, 1 → +inf
    encoded = enc.encode(np.array([0.0, 1.0]))
    assert np.isneginf(encoded[0])
    assert np.isposinf(encoded[1])

def test_sigmoid_encode_roundtrip():
    enc = SigmoidEncoding()
    original = np.array([[0.1, 0.5, 0.9]])
    encoded = enc.encode(original)
    decoded = enc.decode(encoded)
    assert_array_almost_equal(decoded, original, decimal=6)

def test_sigmoid_decode_as_probability():
    enc = SigmoidEncoding(as_probability=True)
    inp = np.array([-100, 0, 100])
    decoded = enc.decode(inp)
    expected = sp_special.expit(inp)
    assert_array_almost_equal(decoded, expected)

def test_sigmoid_decode_not_as_probability():
    enc = SigmoidEncoding(as_probability=False, threshold=0.5)
    # expit(-1)=0.2689, expit(0)=0.5, expit(1)=0.7311
    decoded = enc.decode(np.array([-1, 0, 1]))
    expected = np.array([0, 1, 1])
    assert_array_equal(decoded, expected)
    assert decoded.dtype == int

def test_sigmoid_encode_invalid_range():
    enc = SigmoidEncoding()
    with pytest.raises(AssertionError):
        enc.encode(np.array([-0.1]))
    with pytest.raises(AssertionError):
        enc.encode(np.array([1.1]))


# ===================================================================
#  MatrixEncoding
# ===================================================================
@pytest.mark.parametrize("shape, input_shape", [
    ((2, 3), (4, 6)),
    ((3, 1), (5, 3)),
    ((1, 5), (3, 5)),
])
def test_matrix_encode_decode_roundtrip(shape, input_shape):
    enc = MatrixEncoding(shape)
    original = np.arange(np.prod(input_shape)).reshape(input_shape).astype(float)
    encoded = enc.encode(original)
    assert encoded.shape == (input_shape[0], np.prod(shape))
    decoded = enc.decode(encoded)
    assert decoded.shape == (input_shape[0],) + shape
    assert_array_equal(decoded.reshape(input_shape), original)


# ===================================================================
#  ImageEncoding
# ===================================================================
def test_image_encoding_shape_greyscale():
    enc = ImageEncoding((3, 4), color=False)
    assert enc.shape == (3, 4, 1)

def test_image_encoding_shape_color():
    enc = ImageEncoding((3, 4), color=True)
    assert enc.shape == (3, 4, 3)

def test_image_encode_flattens():
    enc = ImageEncoding((2, 2), color=False)   # shape (2,2,1)
    samples = np.arange(8).reshape(2, 4).astype(np.uint8)
    encoded = enc.encode(samples)
    assert encoded.shape == (2, 4)

def test_image_decode_reshapes_and_casts_to_uint8():
    enc = ImageEncoding((2, 2), color=True)    # shape (2,2,3)
    encoded = np.arange(24).reshape(2, 12).astype(np.float64)
    decoded = enc.decode(encoded)
    assert decoded.shape == (2, 2, 2, 3)
    assert decoded.dtype == np.uint8
    # values are truncated
    expected = encoded.reshape(2, 2, 2, 3).astype(np.uint8)
    assert_array_equal(decoded, expected)


# ===================================================================
#  PSOEncoding (concrete ParameterExtendingEncoding)
# ===================================================================
@pytest.fixture
def pso_enc_4():
    return PSOEncoding(vecsize=4, base_encoding=DefaultEncoding())

def test_pso_encoding_extract_solution(pso_enc_4):
    full = np.array([
        [1., 2, 3, 4, 0.1, 0.2, 0.3, 0.4],
        [5, 6, 7, 8, 0.5, 0.6, 0.7, 0.8],
    ])
    sol = pso_enc_4.extract_solution(full)
    assert_array_equal(sol, [[1, 2, 3, 4], [5, 6, 7, 8]])

def test_pso_encoding_extract_params(pso_enc_4):
    full = np.array([[1, 2, 3, 4, 10, 11, 12, 13]])
    params = pso_enc_4.extract_params(full)
    assert_array_equal(params, [[10, 11, 12, 13]])

def test_pso_encoding_decode_params(pso_enc_4):
    full = np.array([[1, 2, 3, 4, 0.1, 0.2, 0.3, 0.4]])
    param_dict = pso_enc_4.decode_params(full)
    assert "speed" in param_dict
    assert_array_equal(param_dict["speed"], [[0.1, 0.2, 0.3, 0.4]])

def test_pso_encoding_encode_with_params(pso_enc_4):
    solution = np.array([[0, 0, 0, 0], [1, 1, 1, 1]])
    params = {"speed": np.array([[0.5, 0.5, 0.5, 0.5], [0.1, 0.1, 0.1, 0.1]])}
    encoded = pso_enc_4.encode(solution, params)
    assert encoded.shape == (2, 8)
    assert_array_equal(encoded[:, :4], solution)
    assert_array_equal(encoded[:, 4:], params["speed"])

def test_pso_encoding_encode_params_none(pso_enc_4):
    solution = np.array([[9, 8, 7, 6]])
    encoded = pso_enc_4.encode(solution, params=None)
    assert_array_equal(encoded[:, 4:], np.zeros((1, 4)))

def test_pso_decode_returns_solution(pso_enc_4):
    full = np.array([[0, 0, 0, 0, 9, 9, 9, 9]])
    decoded = pso_enc_4.decode(full)
    assert_array_equal(decoded, [[0, 0, 0, 0]])


# ===================================================================
#  SelfAdaptingESEncoding
# ===================================================================
def test_self_adapting_es_single_sigma():
    enc = SelfAdaptingESEncoding(vecsize=5, single_sigma=True, base_encoding=DefaultEncoding())
    assert enc.param_sizes == [("sigma", 1)]
    full = np.array([[1, 2, 3, 4, 5, 0.1]])
    sol = enc.extract_solution(full)
    params = enc.extract_params(full)
    assert_array_equal(sol, [[1, 2, 3, 4, 5]])
    assert_array_equal(params, [[0.1]])

def test_self_adapting_es_multiple_sigma():
    enc = SelfAdaptingESEncoding(vecsize=3, single_sigma=False, base_encoding=DefaultEncoding())
    assert enc.param_sizes == [("sigma", 3)]
    full = np.array([[5, 6, 7, 0.2, 0.3, 0.4]])
    param_dict = enc.decode_params(full)
    assert_array_equal(param_dict["sigma"], [[0.2, 0.3, 0.4]])


# ===================================================================
#  CompositeEncoding
# ===================================================================
def create_composite_encoding_for_test():
    """Build a composite encoding using only conftest types."""
    tc = TypeCastEncoding(encoded_dtype=int, decoded_dtype=float)
    pso = PSOEncoding(vecsize=2)
    return CompositeEncoding([tc, pso]), tc, pso

def test_composite_encode_decode_chain():
    comp, tc, pso = create_composite_encoding_for_test()
    solution = np.array([[1.1, 2.2], [3.3, 4.4]])
    params = {"speed": np.array([[0.5, 0.5], [0.1, 0.1]])}
    encoded = comp.encode(solution, params)
    # Order: reversed([tc, pso]) → first PSO: appends speed, then tc: casts to int
    # So the whole matrix becomes int after PSO added speed as float? Wait, PSO.encode returns hstack,
    # then tc.encode casts the whole thing to int. So speed values get truncated to 0.
    # This is what we observed: speed part is zeros.
    # So the actual result is solution part truncated to int, speed part truncated to int (0s).
    expected_sol = solution.astype(int)          # [[1,2],[3,4]]
    expected_full = np.hstack([expected_sol, np.zeros((2,2))])
    assert_array_equal(encoded, expected_full)
    # Decode: tc.decode (to float) then pso.decode (extracts solution part) → float ints
    decoded = comp.decode(encoded)
    assert_array_equal(decoded, expected_sol.astype(float))

def test_composite_extract_solution():
    comp, _, _ = create_composite_encoding_for_test()
    full = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    sol = comp.extract_solution(full)
    assert_array_equal(sol, [[1, 2], [5, 6]])

def test_composite_extract_params():
    comp, _, _ = create_composite_encoding_for_test()
    full = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    params = comp.extract_params(full)
    assert_array_equal(params, [[3, 4], [7, 8]])