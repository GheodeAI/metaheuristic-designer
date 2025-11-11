from __future__ import annotations
import numpy as np
from numpy import ndarray
from ..ConstraintHandler import RepareConstraint


class ClipBoundConstraint(RepareConstraint):
    def __init__(self, vecsize, low_lim: float = -100, up_lim: float = 100):
        self.vecsize = vecsize

        if np.ndim(low_lim) < 1:
            low_lim = np.repeat(low_lim, vecsize)
        self.low_lim = low_lim

        if np.ndim(up_lim) < 1:
            up_lim = np.repeat(up_lim, vecsize)
        self.up_lim = up_lim

    def repair_solution(self, vector: ndarray) -> ndarray:
        return np.clip(vector, self.low_lim, self.up_lim)
