from __future__ import annotations
import numpy as np
import random
from ..Initializer import Initializer
from ..Individual import Individual
from ..utils import RAND_GEN


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

    def __init__(
        self, genotype_size, low_lim, up_lim, pop_size=1, encoding=None, dtype=float
    ):
        super().__init__(pop_size, encoding)

        self.genotype_size = genotype_size

        if type(low_lim) in [list, tuple, np.ndarray]:
            if len(low_lim) != genotype_size:
                raise ValueError(
                    f"If low_lim is a sequence it must be of length {genotype_size}."
                )

            self.low_lim = low_lim
        else:
            self.low_lim = np.repeat(low_lim, self.genotype_size)

        if type(up_lim) in [list, tuple, np.ndarray]:
            if len(up_lim) != genotype_size:
                raise ValueError(
                    f"If up_lim is a sequence it must be of length {genotype_size}."
                )

            self.up_lim = up_lim
        else:
            self.up_lim = np.repeat(up_lim, self.genotype_size)

        self.dtype = dtype


class UniformVectorInitializer(UniformInitializer):
    def generate_random(self, objfunc):
        new_vector_float = RAND_GEN.uniform(
            self.low_lim, self.up_lim, size=self.genotype_size
        )
        if self.dtype is int:
            new_vector = np.round(new_vector_float).astype(self.dtype)
        else:
            new_vector = new_vector_float.astype(self.dtype)

        return Individual(objfunc, new_vector, encoding=self.encoding)

    def generate_individual(self, objfunc):
        return self.generate_random(objfunc)


class UniformListInitializer(UniformInitializer):
    def generate_random(self, objfunc):
        new_list = [
            RAND_GEN.uniform(low, up) for low, up in zip(self.low_lim, self.up_lim)
        ]
        return Individual(objfunc, new_list, encoding=self.encoding)

    def generate_individual(self, objfunc):
        return self.generate_random(objfunc)
