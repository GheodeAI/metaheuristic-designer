from __future__ import annotations
import numpy as np
from ..Decoder import Decoder


class MatrixDecoder(Decoder):
    """
    Decoder used to evolve matrices
    """

    def __init__(self, shape):
        self.shape = shape

    def encode(self, phenotype: np.ndarray) -> np.ndarray:
        return np.ndarray.flatten(phenotype)

    def decode(self, genotype: np.ndarray) -> np.ndarray:
        return np.reshape(genotype, self.shape)
