from __future__ import annotations
import numpy as np
from numpy import ndarray
from ..Encoding import Encoding


class ImageEncoding(Encoding):
    """
    Decoder used to evolve images.
    """

    def __init__(self, shape, color=True):
        if len(shape) == 2:
            shape = tuple(shape)
            if color:
                shape = shape + (3,)
            else:
                shape = shape + (1,)

        self.shape = shape
        super().__init__(vectorized=True)

    def encode_func(self, solutions: ndarray) -> ndarray:
        return solutions.reshape(solutions.shape[:1] + (-1,))

    def decode_func(self, population: ndarray) -> ndarray:
        return np.reshape(population, population.shape[:1] + self.shape).astype(np.uint8)
