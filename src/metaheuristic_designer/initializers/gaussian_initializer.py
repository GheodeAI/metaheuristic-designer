"""Initializer that samples from a Gaussian (normal) distribution."""

from __future__ import annotations
from numbers import Integral
import numpy as np
from ..initializer import Initializer


class GaussianInitializer(Initializer):
    """
    Initializer that generates individuals with values drawn from a
    Gaussian (normal) distribution.

    Parameters
    ----------
    dimension : int
        Length of the genotype vector.
    g_mean : float or array
        Mean of the distribution.  If an array is given, it must have
        length `dimension`.
    g_std : float or array
        Standard deviation of the distribution.  If an array is given,
        it must have length `dimension`.
    pop_size : int, optional
        Number of individuals to generate (default 1).
    encoding : Encoding, optional
        Encoding that will be passed to each individual.
    dtype : type, optional
        Desired NumPy dtype of the generated vectors (default ``float``).
    rng : RNGLike, optional
        Random number generator.
    """

    def __init__(self, dimension, g_mean, g_std, population_size=1, encoding=None, dtype=float, rng=None):
        super().__init__(dimension=dimension, population_size=population_size, encoding=encoding, rng=rng)

        if type(g_mean) in [list, tuple, np.ndarray]:
            if len(g_mean) != dimension:
                raise ValueError(f"If g_mean is a sequence it must be of length {dimension}.")

            self.g_mean = g_mean
        else:
            self.g_mean = np.repeat(g_mean, self.dimension)

        if type(g_std) in [list, tuple, np.ndarray]:
            if len(g_std) != dimension:
                raise ValueError(f"If g_std is a sequence it must be of length {dimension}.")

            self.g_std = g_std
        else:
            self.g_std = np.repeat(g_std, self.dimension)

        self.dtype = dtype

    def generate_random(self):
        new_vector_float = self.rng.normal(self.g_mean, self.g_std, size=self.dimension)
        if isinstance(self.dtype, Integral):
            new_vector = np.round(new_vector_float).astype(self.dtype)
        else:
            new_vector = new_vector_float.astype(self.dtype)

        return new_vector
