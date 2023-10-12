from __future__ import annotations
import numpy as np
import random
from ..Initializer import Initializer
from ..Individual import Individual
from ..utils import RAND_GEN


class GaussianInitializer(Initializer):
    """
    Initializer that generates individuals with vectors following a normal distribution.

    Parameters
    ----------
    genotype_size: ndarray
        The dimension of the vectors accepted by the objective function.
    g_mean: ndarray or float
        Mean of the probability distribution used to generate the individuals.
    g_str: ndarray or float
        Standard deviation of the probability distribution used to generate the individuals.
    pop_size: int, optional
        Number of individuals to be generated.
    encoding: Encoding, optional
        Encoding that will be passed to each individual.
    dtype: type, optional
        Data type used in each of the components of the vector in the individual.
    """

    def __init__(
        self, genotype_size, g_mean, g_std, pop_size=1, encoding=None, dtype=float
    ):
        super().__init__(pop_size, encoding)

        self.genotype_size = genotype_size

        if type(g_mean) in [list, tuple, np.ndarray]:
            if len(g_mean) != genotype_size:
                raise ValueError(
                    f"If g_mean is a sequence it must be of length {genotype_size}."
                )

            self.g_mean = g_mean
        else:
            self.g_mean = np.repeat(g_mean, self.genotype_size)

        if type(g_std) in [list, tuple, np.ndarray]:
            if len(g_std) != genotype_size:
                raise ValueError(
                    f"If g_std is a sequence it must be of length {genotype_size}."
                )

            self.g_std = g_std
        else:
            self.g_std = np.repeat(g_std, self.genotype_size)

        self.dtype = dtype


class GaussianVectorInitializer(GaussianInitializer):
    def generate_random(self, objfunc):
        new_vector_float = RAND_GEN.normal(
            self.g_mean, self.g_std, size=self.genotype_size
        )
        if self.dtype is int:
            new_vector = np.round(new_vector_float).astype(self.dtype)
        else:
            new_vector = new_vector_float.astype(self.dtype)

        return Individual(objfunc, new_vector, encoding=self.encoding)

    def generate_individual(self, objfunc):
        return self.generate_random(objfunc)


class GaussianListInitializer(GaussianInitializer):
    def generate_random(self, objfunc):
        new_list = [RAND_GEN.normal(m, s) for m, s in zip(self.g_mean, self.g_std)]
        return Individual(objfunc, new_list, encoding=self.encoding)

    def generate_individual(self, objfunc):
        return self.generate_random(objfunc)
