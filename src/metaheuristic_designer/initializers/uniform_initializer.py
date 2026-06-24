"""Initializer that samples from a uniform distribution."""

from __future__ import annotations
import numpy as np
from ..initializer import Initializer


class UniformInitializer(Initializer):
    """
    Initializer that generates individuals with values drawn from a
    uniform distribution.

    Parameters
    ----------
    dimension : int
        Length of the genotype vector.
    lower_bound : float or array
        Lower bound(s) of the distribution.  If an array is given,
        it must have length `dimension`.
    upper_bound : float or array
        Upper bound(s) of the distribution.  Must match the shape
        of `lower_bound`.
    population_size : int, optional
        Number of individuals to generate (default 1).
    encoding : Encoding, optional
        Encoding that will be passed to each individual.
    dtype : type, optional
        Desired NumPy dtype of the generated vectors (default ``float``).
    rng : RNGLike, optional
        Random number generator.
    """

    def __init__(self, dimension, lower_bound, upper_bound, population_size=1, encoding=None, dtype=float, rng=None):
        super().__init__(dimension=dimension, population_size=population_size, encoding=encoding, rng=rng)

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
        new_vector_float = self.rng.uniform(self.lower_bound, self.upper_bound, size=self.dimension)
        if self.dtype is int:
            new_vector = np.round(new_vector_float).astype(self.dtype)
        else:
            new_vector = new_vector_float.astype(self.dtype)

        return new_vector

    def generate_individual(self):
        return self.generate_random()
