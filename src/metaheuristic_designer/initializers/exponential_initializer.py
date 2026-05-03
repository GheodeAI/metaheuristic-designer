from __future__ import annotations
from numbers import Integral
import numpy as np
from ..initializer import Initializer


class ExponentialInitializer(Initializer):
    """
    Initializer that generates individuals with vectors following an uniform distribution.

    Parameters
    ----------
    genotype_size: ndarray
        The dimension of the vectors accepted by the objective function.
    beta: ndarray or float
        Beta parameter of the exponential distribution
    pop_size: int, optional
        Number of individuals to be generated.
    encoding: Encoding, optional
        Encoding that will be passed to each individual.
    dtype: type, optional
        Data type used in each of the components of the vector in the individual.
    """

    def __init__(self, vecsize, beta, pop_size=1, encoding=None, dtype=float, random_state=None):
        super().__init__(vecsize=vecsize, pop_size=pop_size, encoding=encoding, random_state=random_state)

        self.vecsize = vecsize
        self.beta = beta
        self.dtype = dtype

    def generate_random(self):
        new_vector_float = self.random_state.exponential(self.beta, size=self.vecsize)
        if isinstance(self.dtype, Integral):
            new_vector = np.round(new_vector_float).astype(self.dtype)
        else:
            new_vector = new_vector_float.astype(self.dtype)

        return new_vector

    def generate_individual(self):
        return self.generate_random()
