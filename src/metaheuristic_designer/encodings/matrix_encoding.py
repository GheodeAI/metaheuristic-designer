from __future__ import annotations
from typing import Iterable, Tuple
import numpy as np
from numpy import ndarray
from ..encoding import Encoding
from ..utils import MatrixLike


class MatrixEncoding(Encoding):
    """
    Decoder used to evolve matrices.
    """

    def __init__(self, shape: Tuple[int, int]):
        self.shape = tuple(shape)
        super().__init__(decode_as_array=True)

    def encode(self, solutions: Iterable) -> MatrixLike:
        return solutions.reshape(solutions.shape[:1] + (-1,))

    def decode(self, population: MatrixLike) -> Iterable:
        return np.reshape(population, population.shape[:1] + self.shape)
