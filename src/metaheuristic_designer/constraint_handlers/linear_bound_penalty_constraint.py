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
    lower_bound: float | ndarray, optional
        lower limit of the bounds.
    upper_bound: float | ndarray, optional
        upper limit of the bounds.
    """

    def __init__(self, vecsize, alpha: ScalarLike = 1, lower_bound: ScalarLike | VectorLike = -100, upper_bound: ScalarLike | VectorLike = 100):
        self.vecsize = vecsize
        self.alpha = alpha
        if np.ndim(upper_bound) < 1:
            upper_bound = np.repeat(upper_bound, vecsize)
        self.upper_bound = upper_bound
        if np.ndim(lower_bound) < 1:
            lower_bound = np.repeat(lower_bound, vecsize)
        self.lower_bound = lower_bound
        self.range_lim = self.upper_bound - self.lower_bound

    def penalty(self, population_matrix: MatrixLike) -> VectorLike:
        if np.all(self.upper_bound == self.lower_bound):
            if self.upper_bound.ndim == 0:
                return np.full_like(population_matrix, self.upper_bound)
            return np.tile(self.upper_bound, (population_matrix.shape[0], 1))

        low_bound_diff = np.maximum(self.lower_bound - population_matrix, 0)
        up_bound_diff = np.maximum(population_matrix - self.upper_bound, 0)
        total_violation = np.sum(low_bound_diff + up_bound_diff, axis=1)

        return self.alpha * total_violation
