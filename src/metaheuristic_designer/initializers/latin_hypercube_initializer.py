"""Initializer that implements Latin Hypercube Sampling as an initialization technique."""

from __future__ import annotations
from typing import Optional
import numpy as np

from ..objective_function import ObjectiveFunc
from ..population import Population

from .uniform_initializer import UniformInitializer
from ..initializer import Initializer


class LatinHypercubeInitializer(Initializer):
    """
    Initializer that generates individuals using the Latin Hypercube Sampling (LHS)
    technique, in which values are drawn from a stratified uniform distribution that
    more efficiently samples the search space than naive uniform sampling.

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
        self.fallback = UniformInitializer(dimension, lower_bound, upper_bound, dtype=dtype, rng=rng)

    def generate_random(self):
        return self.fallback.generate_random()

    def generate_population(self, objfunc: ObjectiveFunc, n_individuals: Optional[int] = None) -> Population:
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

        idx_matrix = np.tile(np.arange(n_individuals), reps=(self.dimension, 1)).T
        perm_matrix = self.rng.permuted(idx_matrix, axis=0)
        unif_samples = self.rng.random((n_individuals, self.dimension))
        norm_pop_matrix = (perm_matrix + unif_samples) / n_individuals
        population_matrix = (self.upper_bound - self.lower_bound) * norm_pop_matrix + self.lower_bound

        return Population(objfunc, genotype_matrix=population_matrix, encoding=self.encoding)
