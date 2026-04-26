from __future__ import annotations
from copy import copy
import numpy as np
from numpy import ndarray
from ..constraint_handler import RepairConstraint


class BounceBoundConstraint(RepairConstraint):
    """
    Encodes a bound constraint by bouncing through the bounds, substracting the leftover part of the vector
    that lies outisde the bounds. If the substraction still lies outside the bounds, the leftover part is added,
    substraction and addition are alternated until the solution lies in bounds.

    Parameters
    ----------
    vecsize: int
        size of the input vector (decoded).
    low_lim: float | ndarray, optional
        lower limit of the bounds.
    up_lim: float | ndarray, optional
        upper limit of the bounds.
    """

    def __init__(self, vecsize, low_lim: float = -100, up_lim: float = 100):
        self.vecsize = vecsize
        self.low_lim = low_lim
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
