from __future__ import annotations
import numpy as np
from ..constraint_handler import RepairConstraint
from ..utils import MatrixLike, ScalarLike, VectorLike


class ClipBoundConstraint(RepairConstraint):
    """
    Encodes a bound constraint by clipping solutions to the nearest point in the boundary.

    Parameters
    ----------
    dimension: int
        size of the input vector (decoded).
    lower_bound: float | ndarray, optional
        lower limit of the bounds.
    upper_bound: float | ndarray, optional
        upper limit of the bounds.
    """

    def __init__(self, dimension, lower_bound: ScalarLike | VectorLike = -100, upper_bound: ScalarLike | VectorLike = 100, **kwargs):
        self.dimension = dimension
        self.lower_bound = np.asarray(lower_bound)
        self.upper_bound = np.asarray(upper_bound)
        super().__init__(**kwargs)

    def repair_solutions(self, population_matrix: MatrixLike) -> MatrixLike:
        if np.all(self.upper_bound == self.lower_bound):
            if self.upper_bound.ndim == 0:
                return np.full_like(population_matrix, self.upper_bound)
            return np.tile(self.upper_bound, (population_matrix.shape[0], 1))

        return np.clip(population_matrix, self.lower_bound, self.upper_bound)
