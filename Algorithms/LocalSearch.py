import random
import numpy as np
from ..Individual import Indiv
from ..ParamScheduler import ParamScheduler
from .BaseAlgorithm import BaseAlgorithm

class LocalSearch(BaseAlgorithm):
    """
    Search strtategy example, HillClimbing
    """
    
    def __init__(self, objfunc, perturb_op, params={}, name="LocalSearch"):
        """
        Constructor of the Example search strategy class
        """

        super().__init__(objfunc, name)

        self.perturb_op = perturb_op
        self.iterations = params["iters"] if "iters" in params else 100

    def best_solution(self):
        """
        Gives the best solution found by the algorithm and its fitness
        """

        curr_fitness = self.population[0].fitness
        if self.objfunc.opt == "min":
            curr_fitness *= -1
        return (self.population[0].vector, curr_fitness)

    
    def initialize(self):
        """
        Generates a random population of individuals
        """

        self.population[0] = Indiv(self.objfunc, self.objfunc.random_solution())
        self.best_indiv = self.population[0]
        self.best_fit = self.population[0].fitness


    def perturb(self, indiv_list, progress=0, history=None):
        result = []

        for indiv in indiv_list:
            best_indiv = indiv
            
            for i in range(self.iterations):
                # Perturb individual
                new_solution = self.perturb_op(indiv, indiv_list, self.objfunc)
                new_solution = self.objfunc.check_bounds(new_solution)
                new_indiv = Indiv(self.objfunc, new_solution)

                # If it improves the previous solution keep it
                if new_indiv.fitness > best_indiv.fitness:
                    best_indiv = new_indiv
            
            result.append(best_indiv)
        
        return result

    
    def update_params(self, progress):
        """
        Updates the parameters of each component of the algorithm
        """

        self.perturb_op.step(progress)

    
    
