import random
import numpy as np
from numba import jit
from copy import deepcopy

from ..Individual import *
from ..ParamScheduler import ParamScheduler


class HillClimb:    
    """
    Search strtategy example, HillClimbing
    """
    
    def __init__(self, objfunc, perturb_op, params):
        """
        Constructor of the Example search strategy class
        """

        self.params = params

        self.best_fit = 0        

        self.current_indiv = None
        self.best_indiv = None
        self.objfunc = objfunc
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

    
    def step(self, progress, history):
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
        
        self._update_params(progress)
        
        return self.best_solution()

    
    def _update_params(self, progress):
        """
        Updates the parameters of each component of the algorithm
        """

        self.perturb_op.step(progress)

        if isinstance(self.params, ParamScheduler):
            self.params.step(progress)
            self.size = self.params["popSize"]
            self.n_offspring = self.params["offspringSize"]

    
    
