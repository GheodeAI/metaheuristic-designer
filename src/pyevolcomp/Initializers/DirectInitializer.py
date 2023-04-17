from __future__ import annotations
import numpy as np
import random
from ..Initializer import Initializer
from ..Individual import Individual


class DirectInitializer(Initializer):    
    def __init__(self, solutions: List[Individual]):
        self.solutions = solutions
    
    def generate_individual(self, objfunc):
        return random.choice(self.solutions)

    def generate_population(self, objfunc, n_indiv):
        return self.solutions


