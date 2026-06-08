"""Initializer that uses a set of predefined solutions as the first generation."""

from __future__ import annotations
from copy import copy
from typing import Iterable, List, Optional
import numpy as np
from ..objective_function import ObjectiveFunc
from ..initializer import Initializer
from ..population import Population
from ..encoding import Encoding
from ..utils import RNGLike, VectorLike


class DirectInitializer(Initializer):
    """
    Initializer that seeds the population with a given set of solutions.

    If the number of individuals requested exceeds the size of the stored
    set, individuals are cycled through.  Random individuals from a fallback
    initializer are used when :meth:`generate_random` is called directly.

    Parameters
    ----------
    default_init : Initializer
        Fallback initializer for :meth:`generate_random`.
    solutions : Population, list or ndarray
        The set of solutions to draw from.
    encoding : Encoding, optional
        Encoding attached to the population (used when *solutions* is
        a ``Population``).
    rng : RNGLike, optional
        Random number generator.
    """

    def __init__(
        self, default_init: Initializer, solutions: Population | List | np.ndarray, encoding: Encoding = None, rng: Optional[RNGLike] = None
    ):
        assert len(solutions) > 0, "The solution set should not be empty."
        if isinstance(solutions, Population):
            inferred_dimension = solutions.genotype_matrix.shape[1]
        elif isinstance(solutions, Iterable):
            solutions = np.asarray(solutions)
            inferred_dimension = solutions.shape[1]
        else:
            raise TypeError("The provided population is not valid. It should be of type Population, numpy array or list of arrays.")

        self.solutions = solutions
        self.default_init = default_init
        self.init_counter = 0

        super().__init__(dimension=inferred_dimension, population_size=len(solutions), rng=rng, encoding=encoding)

    def generate_random(self) -> VectorLike:
        """Return a completely random individual generated from a fallback initializer strategy

        Returns
        -------
        VectorLike
            A 1-D array sampled from a fallback distribution.
        """

        return self.default_init.generate_random()

    def generate_individual(self) -> VectorLike:
        """Return a chosen individual from the stored solution set in cyclic order.

        Returns
        -------
        VectorLike
            A 1-D array taken from the predefined solutions.
        """

        if isinstance(self.solutions, Population):
            population_matrix = self.solutions.genotype_matrix
        else:
            population_matrix = self.solutions

        indiv = population_matrix[self.init_counter]
        self.init_counter = (self.init_counter + 1) % len(population_matrix)

        return indiv

    def generate_population(self, objfunc: ObjectiveFunc, n_individuals: Optional[int] = None) -> Population:
        """Create a population by drawing from the stored solutions.

        Parameters
        ----------
        objfunc : ObjectiveFunc
            The objective function.
        n_individuals : int, optional
            Number of individuals to generate.  Defaults to
            :attr:`population_size`.

        Returns
        -------
        Population
            A population built from the predefined solutions.
        """

        if n_individuals is None:
            n_individuals = self.population_size

        if isinstance(self.solutions, Population):
            population_matrix = self.solutions.genotype_matrix
        else:
            population_matrix = self.solutions

        selection_idx = np.arange(n_individuals) % population_matrix.shape[0]
        population = Population(objfunc, population_matrix[selection_idx, :], encoding=self.encoding)

        return population
