from __future__ import annotations
from copy import copy
import numpy as np
from numpy import ndarray
from ..ConstraintHandler import RepareConstraint


class BounceBoundConstraint(RepareConstraint):
    def __init__(self, vecsize, low_lim: float = -100, up_lim: float = 100):
        self.vecsize = vecsize

        if np.ndim(low_lim) < 1:
            low_lim = np.repeat(low_lim, vecsize)
        self.low_lim = low_lim

        if np.ndim(up_lim) < 1:
            up_lim = np.repeat(up_lim, vecsize)
        self.up_lim = up_lim
        self.range_lim = up_lim - low_lim

    def repair_solution(self, vector: ndarray) -> ndarray:
        if np.all(self.up_lim == self.low_lim):
            return self.up_lim

        shifted_vector = vector - self.low_lim
        bounce_times = np.floor_divide(shifted_vector, self.range_lim)
        fixed_solution = np.mod((-1.0) ** bounce_times * shifted_vector, self.range_lim) + self.low_lim

        ouside_bound_mask = (vector < self.low_lim) | (vector > self.up_lim)
        vector = copy(vector)
        vector[ouside_bound_mask] = fixed_solution[ouside_bound_mask]

        return vector
