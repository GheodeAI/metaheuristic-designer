from __future__ import annotations
import numpy as np
from numpy import ndarray
from ..constraint_handler import PenalizeConstraint


class LinearBoundPenaltyConstraint(PenalizeConstraint):
    """
    Encodes a bound constraint by adding a penalty proportional to the distance of the
    solution's distance to the bounds.

    Parameters
    ----------
    vecsize: int
        size of the input vector (decoded).
    alpha: float, optional
        factor to multiply to the penalty before being applied.
    low_lim: float | ndarray, optional
        lower limit of the bounds.
    up_lim: float | ndarray, optional
        upper limit of the bounds.
    """

    def __init__(self, vecsize, alpha=1, low_lim: float = -100, up_lim: float = 100):
        self.vecsize = vecsize
        self.alpha = alpha

        if np.ndim(low_lim) < 1:
            low_lim = np.repeat(low_lim, vecsize)
        self.low_lim = low_lim

        if np.ndim(up_lim) < 1:
            up_lim = np.repeat(up_lim, vecsize)
        self.up_lim = up_lim

    def penalty(self, solution: ndarray) -> ndarray:
        low_bound_diff = solution - self.low_lim
        low_bound_diff[low_bound_diff > 0] = 0

        up_bound_diff = solution - self.up_lim
        up_bound_diff[up_bound_diff < 0] = 0

        return self.alpha * np.sum(np.abs(low_bound_diff - up_bound_diff))
