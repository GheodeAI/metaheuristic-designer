from __future__ import annotations
from copy import copy
from typing import List
import random
import numpy as np
from ..Initializer import Initializer
from ..Population import Population
from ..Encoding import Encoding
from ..utils import RAND_GEN


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
    ):
        super().__init__(len(solutions), encoding=encoding)
        self.solutions = solutions
        self.default_init = default_init

    def generate_random(self):
        return self.default_init.generate_random()

    def generate_individual(self):
        indiv = None
        if isinstance(self.solutions, Population):
            indiv = random.choice(self.solutions.genotype_set)
        elif isinstance(self.solutions, np.ndarray):
            indiv = RAND_GEN.choice(self.solutions)

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
            population = Population(objfunc, self.solutions.genotype_set[selection_idx])

        return population
