"""
Initializer that uses a set of predefined solutions as the first generation.
"""

from __future__ import annotations
from copy import copy
from typing import List, Optional
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
    random_state : RNGLike, optional
        Random number generator.
    """

    def __init__(
        self, default_init: Initializer, solutions: Population | List | np.ndarray, encoding: Encoding = None, random_state: Optional[RNGLike] = None
    ):
        assert len(solutions) > 0, "The solution set should not be empty."
        if isinstance(solutions, Population):
            inferred_dimension = solutions.genotype_matrix.shape[1]
        else:
            inferred_dimension = solutions[0].shape[0]

        super().__init__(dimension=inferred_dimension, population_size=default_init.population_size, random_state=random_state, encoding=encoding)
        self.solutions = solutions
        self.default_init = default_init

    def generate_random(self) -> VectorLike:
        return self.default_init.generate_random()

    def generate_individual(self) -> VectorLike:
        """Return a randomly chosen individual from the stored solution set.

        Returns
        -------
        VectorLike
            A 1-D array taken from the predefined solutions.
        """

        indiv = None
        if isinstance(self.solutions, Population):
            indiv = self.random_state.choice(self.solutions.genotype_matrix, axis=0)
        elif isinstance(self.solutions, np.ndarray):
            indiv = self.random_state.choice(self.solutions, axis=0)
        else:
            raise TypeError("The provided population is not valid. It should be of type Population or numpy array.")

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
            if self.solutions.population_size == n_individuals:
                population = copy(self.solutions)
            else:
                selection_idx = np.arange(n_individuals) % self.solutions.population_size
                population = self.solutions.take_selection(selection_idx)
        elif isinstance(self.solutions, np.ndarray):
            selection_idx = np.arange(n_individuals) % self.solutions.shape[0]
            population = Population(objfunc, self.solutions[selection_idx, :])
        else:
            raise TypeError("The provided population is not valid. It should be of type Population or numpy array.")

        return population
