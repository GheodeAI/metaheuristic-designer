from __future__ import annotations
import numpy as np
from numpy import ndarray
from ..encoding import Encoding


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
        super().__init__(decode_as_array=True)

    def encode_func(self, solution: ndarray) -> ndarray:
        return solution.reshape(solution.shape[:1] + (-1,))

    def decode_func(self, population: ndarray) -> ndarray:
        return np.reshape(population, population.shape[:1] + self.shape).astype(np.uint8)
