import random
import numpy as np
from ..Individual import Indiv
from ..ParamScheduler import ParamScheduler
from .BaseAlgorithm import BaseAlgorithm

class HillClimb(BaseAlgorithm):
    """
    Search strtategy example, HillClimbing
    """
    
    def __init__(self, objfunc, perturb_op, name="HillClimb"):
        """
        Constructor of the Example search strategy class
        """

        super().__init__(objfunc, name)

        self.current_indiv = None
        self.best_indiv = None
        self.perturb_op = perturb_op

    def best_solution(self):
        """
        Gives the best solution found by the algorithm and its fitness
        """

        curr_fitness = self.current_indiv.fitness
        if self.objfunc.opt == "min":
            curr_fitness *= -1
        return (self.current_indiv.vector, curr_fitness)

    
    def initialize(self):
        """
        Generates a random population of individuals
        """

        self.current_indiv = Indiv(self.objfunc, self.objfunc.random_solution())
        self.best_fit = self.current_indiv.fitness

    
    def step(self, progress=0, history=None):
        """
        Performs a step of the algorithm
        """

        # Perturb individual
        new_solution = self.perturb_op(self.current_indiv, [self.current_indiv], self.objfunc)
        new_solution = self.objfunc.check_bounds(new_solution)
        new_indiv = Indiv(self.objfunc, new_solution)

        # If it improves the previous solution keep it
        if new_indiv.fitness > self.current_indiv.fitness:
            self.current_indiv = new_indiv        
        
        self.update_params(progress)
        
        return self.best_solution()

    
    def update_params(self, progress):
        """
        Updates the parameters of each component of the algorithm
        """

        self.perturb_op.step(progress)

    
    
