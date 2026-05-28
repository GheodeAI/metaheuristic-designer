"""
Encoding for image-based optimization tasks.
"""

from __future__ import annotations
from typing import Iterable, Tuple
import numpy as np
from ..encoding import Encoding
from ..utils import MatrixLike


class ImageEncoding(Encoding):
    """
    Encoding that maps between flat genotype vectors and image tensors.

    Each individual is reshaped to ``(height, width, channels)``.  When
    ``color`` is ``False`` the channel dimension is omitted (grayscale).

    Parameters
    ----------
    shape : tuple of int
        ``(height, width)`` of the image.
    color : bool, optional
        If ``True`` (default), the image has 3 colour channels (RGB).
        If ``False``, it has 1 channel (grayscale).
    **kwargs
        Forwarded to :class:`Encoding`.
    """

    def __init__(self, shape: Tuple[int, int], color: bool = True, **kwargs):
        if len(shape) == 2:
            shape = tuple(shape)
            if color:
                shape = shape + (3,)
            else:
                shape = shape + (1,)

        self.shape = shape
        super().__init__(decode_as_array=True, **kwargs)

    def encode(self, solution: Iterable) -> MatrixLike:
        return solution.reshape(solution.shape[:1] + (-1,))

    def decode(self, population: MatrixLike) -> Iterable:
        return np.reshape(population, population.shape[:1] + self.shape).astype(np.uint8)
