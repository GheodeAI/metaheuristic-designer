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
    low_lim: ndarray or float
        Lower limit restriction for the vectors.
    up_lim: ndarray or float
        Upper limit restriction for the vectors.
    pop_size: int, optional
        Number of individuals to be generated.
    encoding: Encoding, optional
        Encoding that will be passed to each individual.
    dtype: type, optional
        Data type used in each of the components of the vector in the individual.
    """

    def __init__(self, vecsize, low_lim, up_lim, pop_size=1, encoding=None, dtype=float, random_state=None):
        super().__init__(vecsize=vecsize, pop_size=pop_size, encoding=encoding, random_state=random_state)

        if type(low_lim) in [list, tuple, np.ndarray]:
            if len(low_lim) != vecsize:
                raise ValueError(f"If low_lim is a sequence it must be of length {vecsize}.")

            self.low_lim = low_lim
        else:
            self.low_lim = np.repeat(low_lim, self.vecsize)

        if type(up_lim) in [list, tuple, np.ndarray]:
            if len(up_lim) != vecsize:
                raise ValueError(f"If up_lim is a sequence it must be of length {vecsize}.")

            self.up_lim = up_lim
        else:
            self.up_lim = np.repeat(up_lim, self.vecsize)

        self.dtype = dtype

    def generate_random(self):
        new_vector_float = self.random_state.uniform(self.low_lim, self.up_lim, size=self.vecsize)
        if self.dtype is int:
            new_vector = np.round(new_vector_float).astype(self.dtype)
        else:
            new_vector = new_vector_float.astype(self.dtype)

        return new_vector

    def generate_individual(self):
        return self.generate_random()
