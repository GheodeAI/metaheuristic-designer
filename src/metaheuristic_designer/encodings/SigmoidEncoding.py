from __future__ import annotations
import numpy as np
from numpy import ndarray
from ..Encoding import Encoding


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
        If set to False, returns a boolean set to 1 when :math:`\\sigma(x)` is bigger than a theshold
    threshold: float
        When using `as_probability`, sets the limit at which the value is considered to be a 1.
    """

    def __init__(self, as_probability=True, threshold=0.5):
        assert as_probability or 0 < threshold < 1, "The threshold must be a number between 0 and 1"

        self.as_probability = as_probability
        self.threshold = threshold

        super().__init__(vectorized=True, decode_as_array=True)

    def encode_func(self, solutions: ndarray) -> ndarray:
        if not self.as_probability:
            return solutions
        assert np.all((solutions > 0) & (solutions < 1)), "To encode solutions with the sigmoid encoding, the values must be in the range (0,1)."
        return np.log(solutions / (1 - solutions))

    def decode_func(self, population: ndarray) -> ndarray:
        result = 1 / (1 + np.exp(-population))
        if not self.as_probability:
            result = (result < self.threshold).astype(int)

        return result
