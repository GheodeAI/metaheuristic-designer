from __future__ import annotations
import numpy as np
import random
from ..Initializer import Initializer
from ..Individual import Individual


class SeedProbInitializer(Initializer):
    """
    Initializer that inserts predefined solutions into the population with a given probability, generating
    random individuals otherwise.

    Parameters
    ----------
    default_init: Initializer
        Initializer used to generate random individuals.
    solutions: List[Individual]
        The predefined individuals that will be inserted into the population.
    insert_prob: float
        Probability of inserting one of the predefined solutions into the population.
    """

    def __init__(
        self,
        default_init: Initializer,
        solutions: List[Individual],
        insert_prob: float = 0.1,
    ):
        super().__init__(default_init.pop_size)

        self.default_init = default_init
        self.solutions = solutions
        self.insert_prob = insert_prob

    def generate_random(self, objfunc):
        return self.default_init.generate_random(objfunc)

    def generate_individual(self, objfunc):
        new_indiv = None
        if random.random() < self.insert_prob:
            new_indiv = random.choice(self.solutions)
        else:
            new_indiv = self.default_init.generate_individual(objfunc)
        return new_indiv


class SeedDetermInitializer(Initializer):
    """
    Initializer that inserts predefined solutions into the population with a given probability, generating
    random individuals otherwise.

    Parameters
    ----------
    default_init: Initializer
        Initializer used to generate random individuals.
    solutions: List[Individual]
        The predefined individuals that will be inserted into the population.
    n_to_insert: float
        Amount of predefined individuals to insert in the population.
    """

    def __init__(
        self,
        default_init: Initializer,
        solutions: List[Individual],
        n_to_insert: int = None,
    ):
        super().__init__(default_init.pop_size)

        self.default_init = default_init
        self.solutions = solutions

        self.number_to_insert = n_to_insert
        if n_to_insert is None:
            self.number_to_insert = len(solutions)

        self.inserted = 0

    def generate_random(self, objfunc):
        return self.default_init.generate_random(objfunc)

    def generate_individual(self, objfunc):
        new_indiv = None
        if self.inserted < self.number_to_insert:
            new_indiv = self.solutions[self.inserted % len(self.solutions)]
        else:
            new_indiv = self.default_init.generate_individual(objfunc)

        self.inserted += 1
        return new_indiv

    def generate_population(self, objfunc, n_indiv=None):
        self.inserted = 0

        if n_indiv is None:
            n_indiv = self.pop_size

        return super().generate_population(objfunc, n_indiv)
