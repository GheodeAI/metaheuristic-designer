import random
import numpy as np
from typing import Union
from ..Individual import Indiv
from ..ParamScheduler import ParamScheduler
from ..Operators import Operator
from .BaseAlgorithm import BaseAlgorithm

class LocalSearch(BaseAlgorithm):
    """
    Search strtategy example, HillClimbing
    """
    
    def __init__(self, perturb_op: Operator, params: Union[ParamScheduler, dict]={}, name: str="LocalSearch"):
        """
        Constructor of the Example search strategy class
        """

        super().__init__(name)

        self.population = [None]
        self.perturb_op = perturb_op
        self.iterations = params["iters"] if "iters" in params else 100

    def best_solution(self):
        """
        Gives the best solution found by the algorithm and its fitness
        """

        curr_fitness = self.population[0].fitness
        if self.population[0].objfunc.opt == "min":
            curr_fitness *= -1
        return (self.population[0].vector, curr_fitness)

    
    def initialize(self, objfunc):
        """
        Generates a random population of individuals
        """

        self.population[0] = Indiv(objfunc, objfunc.random_solution())


    def perturb(self, indiv_list, objfunc, progress=0, history=None):
        indiv = indiv_list[0]
        best_indiv = indiv
        for i in range(self.iterations):

            # Perturb individual
            new_indiv = self.perturb_op(indiv, self.population, objfunc, best_indiv)
            new_indiv.vector = objfunc.repair_solution(new_indiv.vector)

            # If it improves the previous solution keep it
            if new_indiv.fitness > best_indiv.fitness:
                best_indiv = new_indiv
        
        return [best_indiv]

    
    def update_params(self, progress):
        """
        Updates the parameters of each component of the algorithm
        """

        self.perturb_op.step(progress)

    
    
