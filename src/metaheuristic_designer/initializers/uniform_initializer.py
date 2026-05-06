from __future__ import annotations
import numpy as np
from ..initializer import Initializer


class UniformInitializer(Initializer):
    """
    Initializer that generates individuals with vectors following an uniform distribution.

    Parameters
    ----------
    genotype_size: ndarray
        The dimension of the vectors accepted by the objective function.
    lower_bound: ndarray or float
        Lower limit restriction for the vectors.
    upper_bound: ndarray or float
        Upper limit restriction for the vectors.
    pop_size: int, optional
        Number of individuals to be generated.
    encoding: Encoding, optional
        Encoding that will be passed to each individual.
    dtype: type, optional
        Data type used in each of the components of the vector in the individual.
    """

    def __init__(self, dimension, lower_bound, upper_bound, pop_size=1, encoding=None, dtype=float, random_state=None):
        super().__init__(dimension=dimension, population_size=pop_size, encoding=encoding, random_state=random_state)

        if type(lower_bound) in [list, tuple, np.ndarray]:
            if len(lower_bound) != dimension:
                raise ValueError(f"If lower_bound is a sequence it must be of length {dimension}.")

            self.lower_bound = lower_bound
        else:
            self.lower_bound = np.repeat(lower_bound, self.dimension)

        if type(upper_bound) in [list, tuple, np.ndarray]:
            if len(upper_bound) != dimension:
                raise ValueError(f"If upper_bound is a sequence it must be of length {dimension}.")

            self.upper_bound = upper_bound
        else:
            self.upper_bound = np.repeat(upper_bound, self.dimension)

        self.dtype = dtype

    def generate_random(self):
        new_vector_float = self.random_state.uniform(self.lower_bound, self.upper_bound, size=self.dimension)
        if self.dtype is int:
            new_vector = np.round(new_vector_float).astype(self.dtype)
        else:
            new_vector = new_vector_float.astype(self.dtype)

        return new_vector

    def generate_individual(self):
        return self.generate_random()
