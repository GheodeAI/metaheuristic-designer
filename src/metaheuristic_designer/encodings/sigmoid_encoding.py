from __future__ import annotations
from typing import Iterable
import scipy as sp
import numpy as np
from numpy import ndarray
from ..encoding import Encoding
from ..utils import MatrixLike


class SigmoidEncoding(Encoding):
    """
    Encoding designed to use optimization algorithms for binary encoded problems
    using algorithms designed for continuous functions.

    Applies the following function to each component of the solution vector:

    :math:`\\sigma(x) = \\frac{1}{1+e^{-x}}`

    Parameters
    ----------
    as_probability: boolean
        If set to True, return a real number in the range (0,1)
        If set to False, returns a boolean set to 1 when :math:`\\sigma(x)` is bigger than a threshold
    threshold: float
        When using `as_probability`, sets the limit at which the value is considered to be a 1.
    """

    def __init__(self, as_probability: bool = True, threshold: float = 0.5):
        assert as_probability or 0 < threshold < 1, "The threshold must be a number between 0 and 1"

        self.as_probability = as_probability
        self.threshold = threshold

        super().__init__(decode_as_array=True)

    def encode(self, solutions: Iterable) -> MatrixLike:
        assert np.all((solutions >= 0) & (solutions <= 1)), "To encode solutions with the sigmoid encoding, the values must be in the range (0,1)."

        mask_zeros = solutions == 0
        mask_ones = solutions == 1
        result = np.log(1 - solutions) - np.log(solutions)
        print(result)
        result[mask_zeros] = -np.inf
        result[mask_ones] = np.inf
        print(result)
        return result

    def decode(self, population: MatrixLike) -> Iterable:
        result = sp.special.expit(population)
        if not self.as_probability:
            result = (result < self.threshold).astype(int)

        return result
