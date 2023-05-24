from __future__ import annotations
import numpy as np
import random
from ..Initializer import Initializer
from ..Individual import Individual


class DirectInitializer(Initializer):    
    def __init__(self, default_init: Initializer, solutions: List[Individual], encoding: Encoding = None):
        super().__init__(len(solutions), encoding=encoding)
        self.solutions = solutions
        self.default_init = default_init
    
    def generate_random(self, objfunc):
        return self.default_init.generate_random(objfunc)
    
    def generate_individual(self, objfunc):
        return random.choice(self.solutions)

    def generate_population(self, objfunc, n_indiv=None):
        return self.solutions[:self.pop_size]


