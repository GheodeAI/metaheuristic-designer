from __future__ import annotations
import numpy as np
from numpy import ndarray
from ..constraint_handler import RepairConstraint


class ClipBoundConstraint(RepairConstraint):
    """
    Encodes a bound constraint by clipping solutions to the nearest point in the boundary.

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
        self.low_lim = np.asarray(low_lim)
        self.up_lim = np.asarray(up_lim)

    def repair_solution(self, solution: ndarray) -> ndarray:
        if np.all(self.up_lim == self.low_lim):
            if self.up_lim.ndim == 1:
                return np.full_like(self.up_lim)
            else:
                return np.full(self.vecsize, self.up_lim)
        return np.clip(solution, self.low_lim, self.up_lim)
