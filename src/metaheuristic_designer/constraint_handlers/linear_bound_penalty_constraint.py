from __future__ import annotations
import numpy as np
from ..constraint_handler import PenalizeConstraint
from ..utils import ScalarLike, VectorLike, MatrixLike


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

    def __init__(self, vecsize, alpha: ScalarLike = 1, low_lim: ScalarLike | VectorLike = -100, up_lim: ScalarLike | VectorLike = 100):
        self.vecsize = vecsize
        self.alpha = alpha
        if np.ndim(up_lim) < 1:
            up_lim = np.repeat(up_lim, vecsize)
        self.up_lim = up_lim
        if np.ndim(low_lim) < 1:
            low_lim = np.repeat(low_lim, vecsize)
        self.low_lim = low_lim
        self.range_lim = self.up_lim - self.low_lim

    def penalty(self, population_matrix: MatrixLike) -> VectorLike:
        if np.all(self.up_lim == self.low_lim):
            if self.up_lim.ndim == 0:
                return np.full_like(population_matrix, self.up_lim)
            return np.tile(self.up_lim, (population_matrix.shape[0], 1))

        low_bound_diff = np.maximum(self.low_lim - population_matrix, 0)
        up_bound_diff = np.maximum(population_matrix - self.up_lim, 0)
        total_violation = np.sum(low_bound_diff + up_bound_diff, axis=1)

        return self.alpha * total_violation
