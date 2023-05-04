from __future__ import annotations
import numpy as np
import random
from ..Initializer import Initializer
from ..Individual import Individual


class DirectInitializer(Initializer):    
    def __init__(self, solutions: List[Individual], encoding: Encoding = None):
        self.solutions = solutions
        super().__init__(len(solutions), encoding=encoding)
    
    def generate_individual(self, objfunc):
        return random.choice(self.solutions)

    def generate_population(self, objfunc, n_indiv=None):
        return self.solutions


