from __future__ import annotations
from copy import copy
import numpy as np
from ..constraint_handler import RepairConstraint
from ..utils import ScalarLike, VectorLike, MatrixLike


class CycleBoundConstraint(RepairConstraint):
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

    def __init__(self, vecsize, low_lim: ScalarLike | VectorLike = -100, up_lim: ScalarLike | VectorLike = 100):
        self.vecsize = vecsize
        self.low_lim = np.asarray(low_lim)
        self.up_lim = np.asarray(up_lim)
        self.range_lim = self.up_lim - self.low_lim

    def repair_solution(self, population_matrix: MatrixLike) -> MatrixLike:
        if np.all(self.up_lim == self.low_lim):
            if self.up_lim.ndim == 0:
                return np.full_like(population_matrix, self.up_lim)
            return np.tile(self.up_lim, (population_matrix.shape[0], 1))

        fixed_solution = np.mod(population_matrix - self.low_lim, self.range_lim) + self.low_lim

        ouside_bound_mask = (population_matrix < self.low_lim) | (population_matrix > self.up_lim)
        population_matrix = copy(population_matrix)
        population_matrix[ouside_bound_mask] = fixed_solution[ouside_bound_mask]

        return population_matrix
