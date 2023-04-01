import pytest

import numpy as np
from pyevolcomp import Individual
from pyevolcomp.Decoders import MatrixDecoder, ImageDecoder, DefaultDecoder

@pytest.mark.parametrize(
    "genotype, phenotype", [
        (1,1),
        ([1,2,3],[1,2,3]),
        (np.array([1,2,3,4]), np.array([1,2,3,4])),
        ([2,[3,4],[[5,6],[7,8],9]], [2,[3,4],[[5,6],[7,8],9]])
    ]
)
def test_default(genotype, phenotype):
    decoder = DefaultDecoder()

    if isinstance(genotype, np.ndarray):
        np.testing.assert_array_equal(decoder.decode(genotype), phenotype)
        np.testing.assert_array_equal(decoder.encode(phenotype), genotype)
    else:
        assert decoder.decode(genotype) == phenotype
        assert decoder.encode(phenotype) == genotype 
        


example = np.random.random([30,40])
@pytest.mark.parametrize(
    "genotype, phenotype", [
        (np.array([1,2,3,4]), np.array([[1,2],[3,4]])),
        (np.ones([100]), np.ones([10,10])),
        (np.ones([200]), np.ones([10,20])),
        (example.flatten(), example),
    ]
)
def test_matrix(genotype, phenotype):
    decoder = MatrixDecoder(phenotype.shape)

    np.testing.assert_array_equal(decoder.decode(genotype), phenotype)
    np.testing.assert_array_equal(decoder.encode(phenotype), genotype)


example_img1 = np.random.randint(0,256,[30,40,1])
@pytest.mark.parametrize(
    "genotype, phenotype", [
        (np.array([1,2,3,4]), np.array([[[1],[2]],[[3],[4]]])),
        (np.ones([100]), np.ones([10,10,1])),
        (np.ones([200]), np.ones([10,20,1])),
        (example_img1.flatten(), example_img1),
    ]
)
def test_gray_img(genotype, phenotype):
    decoder = ImageDecoder(phenotype.shape[:2], color=False)

    np.testing.assert_array_equal(decoder.decode(genotype), phenotype)
    np.testing.assert_array_equal(decoder.encode(phenotype), genotype)


example_img2 = np.random.randint(0,256,[30,40,3])
@pytest.mark.parametrize(
    "genotype, phenotype", [
        (np.array([1,2,3,4,5,6,7,8,9,10,11,12]), np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])),
        (np.ones([300]), np.ones([10,10,3])),
        (np.ones([600]), np.ones([10,20,3])),
        (example_img2.flatten(), example_img2),
    ]
)
def test_rgb_img(genotype, phenotype):
    print(genotype.shape)
    decoder = ImageDecoder(phenotype.shape[:2], color=True)

    np.testing.assert_array_equal(decoder.decode(genotype), phenotype)
    np.testing.assert_array_equal(decoder.encode(phenotype), genotype)
    
