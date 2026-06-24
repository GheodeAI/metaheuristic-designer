from __future__ import annotations
from copy import copy
import numpy as np
from ..constraint_handler import RepairConstraint
from ..utils import MatrixLike, ScalarLike, VectorLike


class BounceBoundConstraint(RepairConstraint):
    """
    Encodes a bound constraint by bouncing through the bounds, subtracting the leftover part of the vector
    that lies outside the bounds. If the subtraction still lies outside the bounds, the leftover part is added,
    subtraction and addition are alternated until the solution lies in bounds.

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
        self.range_lim = self.upper_bound - self.lower_bound
        super().__init__(**kwargs)

    def repair_solutions(self, population_matrix: MatrixLike) -> MatrixLike:
        if np.all(self.upper_bound == self.lower_bound):
            if self.upper_bound.ndim == 0:
                return np.full_like(population_matrix, self.upper_bound)
            return np.tile(self.upper_bound, (population_matrix.shape[0], 1))

        shifted_vector = population_matrix - self.lower_bound
        bounce_times = np.floor_divide(shifted_vector, self.range_lim)
        fixed_solution = np.mod((-1.0) ** bounce_times * shifted_vector, self.range_lim) + self.lower_bound

        outside_bound_mask = (population_matrix < self.lower_bound) | (population_matrix > self.upper_bound)
        population_matrix = copy(population_matrix)
        population_matrix[outside_bound_mask] = fixed_solution[outside_bound_mask]

        return population_matrix
