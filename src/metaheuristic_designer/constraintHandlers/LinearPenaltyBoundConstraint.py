from __future__ import annotations
import numpy as np
from numpy import ndarray
from ..ConstraintHandler import PenalizeConstraint


class LinearPenaltyBoundConstraint(PenalizeConstraint):
    def __init__(self, vecsize, alpha=1, low_lim: float = -100, up_lim: float = 100):
        self.vecsize = vecsize
        self.alpha = alpha

        if np.ndim(low_lim) < 1:
            low_lim = np.repeat(low_lim, vecsize)
        self.low_lim = low_lim

        if np.ndim(up_lim) < 1:
            up_lim = np.repeat(up_lim, vecsize)
        self.up_lim = up_lim

    def penalty(self, vector: ndarray) -> ndarray:
        low_bound_diff = vector - self.low_lim
        low_bound_diff[low_bound_diff > 0] = 0

        up_bound_diff = vector - self.up_lim
        up_bound_diff[up_bound_diff < 0] = 0

        return self.alpha * np.sum(np.abs(low_bound_diff - up_bound_diff))
