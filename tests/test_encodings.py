import pytest

import numpy as np
from metaheuristic_designer import Individual
from metaheuristic_designer.encodings import (
    MatrixEncoding,
    ImageEncoding,
    DefaultEncoding,
    TypeCastEncoding,
)
import metaheuristic_designer as mhd

mhd.reset_seed(0)


@pytest.mark.parametrize(
    "genotype, phenotype",
    [
        (1, 1),
        ([1, 2, 3], [1, 2, 3]),
        (np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4])),
        ([2, [3, 4], [[5, 6], [7, 8], 9]], [2, [3, 4], [[5, 6], [7, 8], 9]]),
    ],
)
def test_default(genotype, phenotype):
    encoding = DefaultEncoding()

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
            np.array([1, 2, 6, 4, 6], dtype=int),
            np.array([1, 2, 6, 4, 6], dtype=int),
            int,
            int,
        ),
        (
            np.array([1.5, 2.2, 6.1, 4.4, 6.2], dtype=float),
            np.array([1.5, 2.2, 6.1, 4.4, 6.2], dtype=float),
            float,
            float,
        ),
        (
            np.array([1.5, 2.2, 6.1, 4.4, 6.2], dtype=float),
            np.array([1, 2, 6, 4, 6], dtype=int),
            float,
            int,
        ),
        (
            np.array([1, 2, 6, 4, 6], dtype=int),
            np.array([2.0, 2.0, 6.0, 4.0, 6.0], dtype=float),
            int,
            float,
        ),
        (
            np.array([0, 1, 1, 0, 0], dtype=int),
            np.array([False, True, True, False, False], dtype=bool),
            int,
            bool,
        ),
        (
            np.array([False, True, True, False, False], dtype=bool),
            np.array([0, 1, 1, 0, 0], dtype=int),
            bool,
            int,
        ),
    ],
)
def test_typecast(genotype, phenotype, type_in, type_out):
    encoding = TypeCastEncoding(type_in, type_out)

    assert encoding.decode(genotype).dtype is np.dtype(type_out)
    assert encoding.encode(phenotype).dtype is np.dtype(type_in)


example = np.random.random([30, 40])


@pytest.mark.parametrize(
    "genotype, phenotype",
    [
        (np.array([1, 2, 3, 4]), np.array([[1, 2], [3, 4]])),
        (np.ones([100]), np.ones([10, 10])),
        (np.ones([200]), np.ones([10, 20])),
        (example.flatten(), example),
    ],
)
def test_matrix(genotype, phenotype):
    encoding = MatrixEncoding(phenotype.shape)

    np.testing.assert_array_equal(encoding.decode(genotype), phenotype)
    np.testing.assert_array_equal(encoding.encode(phenotype), genotype)


example_img1 = np.random.randint(0, 256, [30, 40, 1])


@pytest.mark.parametrize(
    "genotype, phenotype",
    [
        (np.array([1, 2, 3, 4]), np.array([[[1], [2]], [[3], [4]]])),
        (np.ones([100]), np.ones([10, 10, 1])),
        (np.ones([200]), np.ones([10, 20, 1])),
        (example_img1.flatten(), example_img1),
    ],
)
def test_gray_img(genotype, phenotype):
    encoding = ImageEncoding(phenotype.shape[:2], color=False)

    np.testing.assert_array_equal(encoding.decode(genotype), phenotype)
    np.testing.assert_array_equal(encoding.encode(phenotype), genotype)


example_img2 = np.random.randint(0, 256, [30, 40, 3])


@pytest.mark.parametrize(
    "genotype, phenotype",
    [
        (
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
            np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]),
        ),
        (np.ones([300]), np.ones([10, 10, 3])),
        (np.ones([600]), np.ones([10, 20, 3])),
        (example_img2.flatten(), example_img2),
    ],
)
def test_rgb_img(genotype, phenotype):
    encoding = ImageEncoding(phenotype.shape[:2], color=True)

    np.testing.assert_array_equal(encoding.decode(genotype), phenotype)
    np.testing.assert_array_equal(encoding.encode(phenotype), genotype)
