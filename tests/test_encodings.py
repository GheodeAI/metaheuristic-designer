import pytest

import numpy as np
from metaheuristic_designer import DefaultEncoding
from metaheuristic_designer.encodings import *
import metaheuristic_designer as mhd


@pytest.mark.parametrize(
    "genotype, phenotype",
    [
        (1, 1),
        ([[1, 2, 3]], [[1, 2, 3]]),
        (np.array([[1, 2, 3, 4]]), np.array([[1, 2, 3, 4]])),
        ([2, [3, 4], [[5, 6], [7, 8], 9]], [2, [3, 4], [[5, 6], [7, 8], 9]]),
    ],
)
def test_default(genotype, phenotype):
    encoding = DefaultEncoding(decode_as_array=isinstance(genotype, np.ndarray))

    if isinstance(genotype, np.ndarray):
        np.testing.assert_array_equal(encoding.decode(genotype), phenotype)
        np.testing.assert_array_equal(encoding.encode(phenotype), genotype)
    else:
        assert encoding.decode(genotype) == phenotype
        assert encoding.encode(phenotype) == genotype


@pytest.mark.parametrize(
    "genotype, phenotype, type_in, type_out",
    [
        (
            np.array([[1, 2, 6, 4, 6]], dtype=int),
            np.array([[1, 2, 6, 4, 6]], dtype=int),
            int,
            int,
        ),
        (
            np.array([[1.5, 2.2, 6.1, 4.4, 6.2]], dtype=float),
            np.array([[1.5, 2.2, 6.1, 4.4, 6.2]], dtype=float),
            float,
            float,
        ),
        (
            np.array([[1.5, 2.2, 6.1, 4.4, 6.2]], dtype=float),
            np.array([[1, 2, 6, 4, 6]], dtype=int),
            float,
            int,
        ),
        (
            np.array([[1, 2, 6, 4, 6]], dtype=int),
            np.array([[1.0, 2.0, 6.0, 4.0, 6.0]], dtype=float),
            int,
            float,
        ),
        (
            np.array([[0, 1, 1, 0, 0]], dtype=int),
            np.array([[False, True, True, False, False]], dtype=bool),
            int,
            bool,
        ),
        (
            np.array([[False, True, True, False, False]], dtype=bool),
            np.array([[0, 1, 1, 0, 0]], dtype=int),
            bool,
            int,
        ),
    ],
)
def test_typecast(genotype, phenotype, type_in, type_out):
    encoding = TypeCastEncoding(type_in, type_out)

    assert encoding.decode(genotype).dtype is np.dtype(type_out)
    assert encoding.encode(phenotype).dtype is np.dtype(type_in)
    np.testing.assert_array_equal(encoding.decode(genotype), phenotype)


example = np.random.random((4, 30, 40))
example_flat = example.reshape((4, 1200))


@pytest.mark.parametrize(
    "genotype, phenotype",
    [
        (np.array([[1, 2, 3, 4]]), np.array([[[1, 2], [3, 4]]])),
        (np.ones((1, 100)), np.ones((1, 10, 10))),
        (np.ones((4, 200)), np.ones((4, 10, 20))),
        (example_flat, example),
    ],
)
def test_matrix(genotype, phenotype):
    encoding = MatrixEncoding(phenotype.shape[1:])

    np.testing.assert_array_equal(encoding.decode(genotype), phenotype)
    np.testing.assert_array_equal(encoding.encode(phenotype), genotype)


example_img1 = np.random.randint(0, 256, (1, 30, 40, 1))
example_img_flat1 = example_img1.reshape((1, 1200))


@pytest.mark.parametrize(
    "genotype, phenotype, shape",
    [
        (np.array([[1, 2, 3, 4]]), np.array([[[[1], [2]], [[3], [4]]]]), (2, 2)),
        (np.ones((1, 100)), np.ones((1, 10, 10, 1)), (10, 10)),
        (np.ones((4, 200)), np.ones((4, 10, 20, 1)), (10, 20)),
        (example_img_flat1, example_img1, example_img1.shape[1:3]),
    ],
)
def test_gray_img(genotype, phenotype, shape):
    encoding = ImageEncoding(shape, color=False)

    np.testing.assert_array_equal(encoding.decode(genotype), phenotype)
    np.testing.assert_array_equal(encoding.encode(phenotype), genotype)


example_img2 = np.random.randint(0, 256, (1, 30, 40, 3))
example_img_flat2 = example_img2.reshape((1, 3600))


@pytest.mark.parametrize(
    "genotype, phenotype, shape",
    [
        (np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]), np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]), (2, 2)),
        (np.ones((1, 300)), np.ones((1, 10, 10, 3)), (10, 10)),
        (np.ones((4, 600)), np.ones((4, 10, 20, 3)), (10, 20)),
        (example_img_flat2, example_img2, example_img2.shape[1:3]),
    ],
)
def test_rgb_img(genotype, phenotype, shape):
    encoding = ImageEncoding(shape, color=True)

    np.testing.assert_array_equal(encoding.decode(genotype), phenotype)
    np.testing.assert_array_equal(encoding.encode(phenotype), genotype)

# ============================================================================
# SigmoidEncoding
# ============================================================================
@pytest.mark.parametrize(
    "genotype, as_prob, threshold, expected_decode",
    [
        # probability mode: sigmoid
        (np.array([[0.0]]), True, 0.5, np.array([[0.5]])), 
        (np.array([[999.0]]), True, 0.5, np.array([[1.0]])),  # high value → 1
        (np.array([[-999.0]]), True, 0.5, np.array([[0.0]])), # low value → 0
        # binary mode with threshold
        (np.array([[0.0]]), False, 0.5, np.array([[False]])),
        (np.array([[1.0]]), False, 0.5, np.array([[True]])),
        (np.array([[-1.0]]), False, 0.5, np.array([[False]])),
    ],
)
def test_sigmoid_decode(genotype, as_prob, threshold, expected_decode):
    enc = SigmoidEncoding(as_probability=as_prob, threshold=threshold)
    decoded = enc.decode(genotype)
    np.testing.assert_array_equal(decoded, expected_decode)


def test_sigmoid_encode():
    enc = SigmoidEncoding(as_probability=True)
    # encode should be the identity (or pass‑through) for this encoding?
    # Check that it returns the input as an array
    inp = np.array([[0.5], [0.2]])
    encoded = enc.encode(inp)
    np.testing.assert_array_equal(encoded, inp)


# ============================================================================
# EncodingFromLambda
# ============================================================================
def test_encoding_from_lambda():
    # simple encode: double the value, decode: halve
    encode_fn = lambda x: 2 * np.asarray(x)
    decode_fn = lambda y: np.asarray(y) / 2.0
    enc = EncodingFromLambda(encode_fn, decode_fn)

    genotype = np.array([[1.0, 2.0], [3.0, 4.0]])
    phenotype = np.array([[0.5, 1.0], [1.5, 2.0]])

    np.testing.assert_array_equal(enc.encode(phenotype), genotype)
    np.testing.assert_array_equal(enc.decode(genotype), phenotype)


# ============================================================================
# CompositeEncoding
# ============================================================================
def test_composite_encoding():
    # Two sub‑encodings:
    #   - part 1: DefaultEncoding (keeps as is)
    #   - part 2: SigmoidEncoding (probability mode)
    part1 = DefaultEncoding(decode_as_array=True)
    part2 = SigmoidEncoding(as_probability=True)

    comp = CompositeEncoding([part1, part2])

    # sample solution: a dict with keys 0 and 1 (index in the list)
    solution = {
        0: np.array([[3.0, 4.0]]),
        1: np.array([[0.0, 10.0]]),  # will be decoded to sigmoid(0)=0.5, sigmoid(10)~1
    }
    # encoded genotype: concatenation of encoded parts
    encoded = comp.encode([solution])  # list of solutions
    # expected: first part unchanged: [[3,4]], second part unchanged (identity encode) [[0,10]]
    expected_encoded = np.array([[3, 4, 0, 10]])
    np.testing.assert_array_equal(encoded, expected_encoded)

    # decoding
    decoded = comp.decode(np.array([[3, 4, 0, 10]]))
    # part1 unchanged: [3,4], part2 decoded via sigmoid: [0.5, ~1.0]
    expected_decoded = [np.array([3.0, 4.0]), np.array([0.5, 0.9999546])]
    assert len(decoded) == 2
    np.testing.assert_array_almost_equal(decoded[0], expected_decoded[0])
    np.testing.assert_array_almost_equal(decoded[1], expected_decoded[1])


# ============================================================================
# ParameterExtendingEncoding (via PSOEncoding)
# ============================================================================
def test_pso_encoding_extracts_solution_and_params():
    # PSOEncoding: vecsize = 5, default base_encoding=None
    enc = PSOEncoding(vecsize=5)
    # an individual with 5 solution genes + 5 speed genes = 10 total
    indiv = np.array([[1, 2, 3, 4, 5, 0.1, 0.2, 0.3, 0.4, 0.5]])

    solution_part = enc.extract_solution(indiv)
    params_part = enc.extract_params(indiv)

    np.testing.assert_array_equal(solution_part, np.array([[1, 2, 3, 4, 5]]))
    np.testing.assert_array_equal(params_part, np.array([[0.1, 0.2, 0.3, 0.4, 0.5]]))


def test_pso_encode_decode_params():
    enc = PSOEncoding(vecsize=5)
    param_dict = {"speed": np.array([0.1, 0.2, 0.3, 0.4, 0.5])}
    encoded = enc.encode_params(param_dict)
    np.testing.assert_array_equal(encoded, np.array([[0.1, 0.2, 0.3, 0.4, 0.5]]))

    decoded = enc.decode_params(encoded)
    np.testing.assert_array_equal(decoded["speed"], np.array([0.1, 0.2, 0.3, 0.4, 0.5]))


def test_pso_full_encode_decode():
    enc = PSOEncoding(vecsize=5)
    solution = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    params = {"speed": np.array([0.1, 0.2, 0.3, 0.4, 0.5])}
    indiv = enc.encode([solution], params=params)
    expected = np.array([[1, 2, 3, 4, 5, 0.1, 0.2, 0.3, 0.4, 0.5]])
    np.testing.assert_array_equal(indiv, expected)

    decoded_solutions = enc.decode(indiv)
    decoded_params = enc.decode_params(indiv)
    np.testing.assert_array_equal(decoded_solutions[0], solution)
    np.testing.assert_array_equal(decoded_params["speed"], np.array([0.1, 0.2, 0.3, 0.4, 0.5]))