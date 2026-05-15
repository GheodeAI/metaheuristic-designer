"""
Initializer that samples from an exponential distribution.
"""

from __future__ import annotations
from numbers import Integral
import numpy as np
from ..initializer import Initializer


class ExponentialInitializer(Initializer):
    """
    Initializer that generates individuals with values drawn from an
    exponential distribution.

    Parameters
    ----------
    dimension : int
        Length of the genotype vector.
    beta : float or array
        Scale parameter of the exponential distribution (1 / rate).
    pop_size : int, optional
        Number of individuals to generate (default 1).
    encoding : Encoding, optional
        Encoding that will be passed to each individual.
    dtype : type, optional
        Desired NumPy dtype of the generated vectors (default ``float``).
    random_state : RNGLike, optional
        Random number generator.
    """

    def __init__(self, dimension, beta, pop_size=1, encoding=None, dtype=float, random_state=None):
        super().__init__(dimension=dimension, population_size=pop_size, encoding=encoding, random_state=random_state)

        self.dimension = dimension
        self.beta = beta
        self.dtype = dtype

    def generate_random(self):
        new_vector_float = self.random_state.exponential(self.beta, size=self.dimension)
        if isinstance(self.dtype, Integral):
            new_vector = np.round(new_vector_float).astype(self.dtype)
        else:
            new_vector = new_vector_float.astype(self.dtype)

        return new_vector

    def generate_individual(self):
        return self.generate_random()
