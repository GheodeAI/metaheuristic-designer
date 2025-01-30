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
        super().__init__(vectorized=True, decode_as_array=True)

    def encode_func(self, solutions: ndarray) -> ndarray:
        return solutions.reshape(solutions.shape[:1] + (-1,))

    def decode_func(self, population: ndarray) -> ndarray:
        return np.reshape(population, population.shape[:1] + self.shape)
