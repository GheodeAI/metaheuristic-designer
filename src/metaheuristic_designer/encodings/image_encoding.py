from __future__ import annotations
from typing import Iterable, Tuple
import numpy as np
from ..encoding import Encoding
from ..utils import MatrixLike


class ImageEncoding(Encoding):
    """
    Decoder used to evolve images.
    """

    def __init__(self, shape: Tuple[int, int], color: bool = True):
        if len(shape) == 2:
            shape = tuple(shape)
            if color:
                shape = shape + (3,)
            else:
                shape = shape + (1,)

        self.shape = shape
        super().__init__(decode_as_array=True)

    def encode(self, solution: Iterable) -> MatrixLike:
        return solution.reshape(solution.shape[:1] + (-1,))

    def decode(self, population: MatrixLike) -> Iterable:
        return np.reshape(population, population.shape[:1] + self.shape).astype(np.uint8)
