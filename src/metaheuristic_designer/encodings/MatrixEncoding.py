from __future__ import annotations
import numpy as np
from numpy import ndarray
from ..Encoding import Encoding


class MatrixEncoding(Encoding):
    """
    Decoder used to evolve matrices.
    """

    def __init__(self, shape):
        self.shape = shape

    def encode(self, phenotype: ndarray) -> ndarray:
        return phenotype.flatten()

    def decode(self, genotype: ndarray) -> ndarray:
        return np.reshape(genotype, self.shape)
