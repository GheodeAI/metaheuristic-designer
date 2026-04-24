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

    def __init__(
        self,
        default_init: Initializer,
        solutions: Population | List | np.ndarray,
        encoding: Encoding = None,
        random_state = None
    ):
        super().__init__(len(solutions), encoding=encoding, random_state=random_state)
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

    def generate_population(self, objfunc, n_indiv=None):
        if n_indiv is None:
            n_indiv = self.pop_size

        if isinstance(self.solutions, Population):
            if self.solutions.pop_size == n_indiv:
                population = copy(self.solutions)
            else:
                selection_idx = np.arange(n_indiv) % self.solutions.pop_size
                population = self.solutions.take_selection(selection_idx)
        elif isinstance(self.solutions, np.ndarray):
            selection_idx = np.arange(n_indiv) % self.solutions.pop_size
            population = Population(objfunc, self.solutions.genotype_matrix[selection_idx])
        else:
            raise TypeError("The provided population is not valid. It should be of type Population or numpy array.")

        return population
