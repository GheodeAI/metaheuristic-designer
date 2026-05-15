"""
Encoding that reshapes vectors into matrices.
"""

from __future__ import annotations
from typing import Iterable, Tuple
import numpy as np
from ..encoding import Encoding
from ..utils import MatrixLike


class MatrixEncoding(Encoding):
    """
    Encoding that reshapes flat genotype vectors into 2-D matrices.

    Each individual is reshaped according to the given *shape*.

    Parameters
    ----------
    shape : tuple of int
        ``(rows, cols)`` of the resulting matrix.
    **kwargs
        Forwarded to :class:`Encoding`.
    """

    def __init__(self, shape: Tuple[int, int], **kwargs):
        self.shape = tuple(shape)
        super().__init__(decode_as_array=True, **kwargs)

    def encode(self, solutions: Iterable) -> MatrixLike:
        return solutions.reshape(solutions.shape[:1] + (-1,))

    def decode(self, population: MatrixLike) -> Iterable:
        return np.reshape(population, population.shape[:1] + self.shape)
