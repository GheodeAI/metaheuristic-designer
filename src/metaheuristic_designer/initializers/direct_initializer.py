from __future__ import annotations
from copy import copy
from typing import List
import numpy as np
from ..initializer import Initializer
from ..population import Population
from ..encoding import Encoding


class DirectInitializer(Initializer):
    """
    Initializer that uses a predefined population to generate the first generation.

    Parameters
    ----------
    default_init: Initializer
        Initializer used to generate random individuals
    solutions: List[Individual]
        The initial population.
    encoding: Encoding, optional
        Encoding that will be passed to each individual.
    """

    def __init__(self, default_init: Initializer, solutions: Population | List | np.ndarray, encoding: Encoding = None, random_state=None):
        assert len(solutions) > 0, "The solution set should not be empty."
        if isinstance(solutions, Population):
            infered_dimension = solutions.genotype_matrix.shape[1]
        else:
            infered_dimension = solutions[0].shape[0]

        super().__init__(dimension=infered_dimension, population_size=default_init.population_size, random_state=random_state)
        self.solutions = solutions
        self.default_init = default_init

    def generate_random(self):
        return self.default_init.generate_random()

    def generate_individual(self):
        indiv = None
        if isinstance(self.solutions, Population):
            indiv = self.random_state.choice(self.solutions.genotype_matrix, axis=0)
        elif isinstance(self.solutions, np.ndarray):
            indiv = self.random_state.choice(self.solutions, axis=0)
        else:
            raise TypeError("The provided population is not valid. It should be of type Population or numpy array.")

        return indiv

    def generate_population(self, objfunc, n_individuals=None):
        if n_individuals is None:
            n_individuals = self.population_size

        if isinstance(self.solutions, Population):
            if self.solutions.pop_size == n_individuals:
                population = copy(self.solutions)
            else:
                selection_idx = np.arange(n_individuals) % self.solutions.pop_size
                population = self.solutions.take_selection(selection_idx)
        elif isinstance(self.solutions, np.ndarray):
            selection_idx = np.arange(n_individuals) % self.solutions.shape[0]
            population = Population(objfunc, self.solutions[selection_idx, :])
        else:
            raise TypeError("The provided population is not valid. It should be of type Population or numpy array.")

        return population
