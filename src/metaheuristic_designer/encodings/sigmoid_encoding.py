"""
Encoding that applies a sigmoid function to enable continuous operators on binary problems.
"""

from __future__ import annotations
from typing import Iterable
import scipy as sp
import numpy as np
from ..encoding import Encoding
from ..utils import MatrixLike


class SigmoidEncoding(Encoding):
    """
    Encoding that maps binary solutions to continuous values via a sigmoid.

    The encoding applies :math:`\\sigma(x) = 1 / (1 + e^{-x})` pointwise.
    During encoding, the logit function is applied to the probability
    parameter (producing real numbers).  During decoding, the sigmoid
    is applied again.  This allows real-valued operators (e.g., Gaussian
    mutation) to be used on binary problems.

    Parameters
    ----------
    as_probability : bool, optional
        If ``True`` (default), each component is returned as a
        probability in (0, 1).  If ``False``, the probability is
        thresholded to produce a hard 0/1 value.
    threshold : float, optional
        Threshold used when ``as_probability=False``.  Must be in
        (0, 1).  Default is 0.5.
    **kwargs
        Forwarded to :class:`Encoding`.
    """

    def __init__(self, as_probability: bool = True, threshold: float = 0.5, **kwargs):
        assert as_probability or 0 < threshold < 1, "The threshold must be a number between 0 and 1"

        self.as_probability = as_probability
        self.threshold = threshold

        super().__init__(decode_as_array=True, **kwargs)

    def encode(self, solutions: Iterable) -> MatrixLike:
        assert np.all((solutions >= 0) & (solutions <= 1)), "To encode solutions with the sigmoid encoding, the values must be in the range (0,1)."

        result = sp.special.logit(solutions)
        return result

    def decode(self, population: MatrixLike) -> Iterable:
        result = sp.special.expit(population)
        if not self.as_probability:
            result = (result >= self.threshold).astype(int)

        return result
