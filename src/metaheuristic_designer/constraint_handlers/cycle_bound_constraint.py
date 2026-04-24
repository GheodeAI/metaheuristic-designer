from __future__ import annotations
from copy import copy
import numpy as np
from numpy import ndarray
from ..constraint_handler import RepareConstraint


class CycleBoundConstraint(RepareConstraint):
    """
    Encodes a bound constraint by wrapping through the bounds, performing a modulo
    operation componentwise.

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

    def repair_solution(self, solution: ndarray) -> ndarray:
        if np.all(self.up_lim == self.low_lim):
            return self.up_lim

        fixed_solution = np.mod(solution - self.low_lim, self.range_lim) + self.low_lim

        ouside_bound_mask = (solution < self.low_lim) | (solution > self.up_lim)
        solution = copy(solution)
        solution[ouside_bound_mask] = fixed_solution[ouside_bound_mask]

        return solution
