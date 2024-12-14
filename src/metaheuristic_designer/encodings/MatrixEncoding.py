from __future__ import annotations
import numpy as np
from numpy import ndarray
from ..Encoding import Encoding


class MatrixEncoding(Encoding):
    """
    Decoder used to evolve matrices.
    """

    def __init__(self, shape):
        self.shape = tuple(shape)

    def encode(self, phenotypes: ndarray) -> ndarray:
        return phenotypes.reshape(phenotypes.shape[:1] + (-1,))

    def decode(self, genotypes: ndarray) -> ndarray:
        return np.reshape(genotypes, genotypes.shape[:1] + self.shape)
