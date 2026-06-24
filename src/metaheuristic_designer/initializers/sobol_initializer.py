"""Initializer that implements Sobol sequences as an initialization technique."""

from __future__ import annotations
from typing import Optional
import numpy as np
import scipy as sp

from ..objective_function import ObjectiveFunc
from ..population import Population

from .uniform_initializer import UniformInitializer
from ..initializer import Initializer


class SobolInitializer(Initializer):
    """
    Initializer that generates individuals using the Sobol sequences,
    this is a quasi-random method designed for covering the space
    with low-discrepancy samples.

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

    def __init__(
        self,
        dimension,
        lower_bound,
        upper_bound,
        population_size=1,
        scramble=True,
        fallback: Optional[Initializer] = None,
        encoding=None,
        dtype=float,
        rng=None,
    ):
        super().__init__(dimension=dimension, population_size=population_size, encoding=encoding, rng=rng)
        self.dtype = dtype

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

        if fallback is None:
            fallback = UniformInitializer(dimension, lower_bound, upper_bound, dtype=dtype, rng=rng)
        self.fallback = fallback
        self.scramble = scramble

    def generate_random(self):
        return self.fallback.generate_random()

    def generate_population(self, n_individuals: Optional[int] = None) -> Population:
        """
        Create a fully formed population of *n_individuals* individuals.

        Parameters
        ----------
        objfunc: ObjectiveFunc
            Objective function that will be propagated to each individual.
        n_individual: int, optional
            Number of individuals to generate

        Returns
        -------
        generated_population: Population
            Newly generated population.
        """

        if n_individuals is None:
            n_individuals = self.population_size

        n_bits = int(np.ceil(np.log2(n_individuals)))
        generator = sp.stats.qmc.Sobol(d=self.dimension, scramble=self.scramble, rng=self.rng)
        samples = generator.random_base2(n_bits)[:n_individuals]
        population_matrix = (self.upper_bound - self.lower_bound) * samples + self.lower_bound

        return Population(genotype_matrix=population_matrix, encoding=self.encoding)
