import random
import numpy as np
from ..Individual import Indiv
from ..ParamScheduler import ParamScheduler
from .BaseAlgorithm import BaseAlgorithm


class HillClimb(BaseAlgorithm):
    """
    Search strtategy example, HillClimbing
    """
    
    def __init__(self, perturb_op, name="HillClimb"):
        """
        Constructor of the Example search strategy class
        """

        super().__init__(name)

        self.population = [None]
        self.perturb_op = perturb_op

    def best_solution(self, objfunc):
        """
        Gives the best solution found by the algorithm and its fitness
        """

        curr_fitness = self.population[0].fitness
        if objfunc.opt == "min":
            curr_fitness *= -1
        return (self.population[0].vector, curr_fitness)

    
    def initialize(self, objfunc):
        """
        Generates a random population of individuals
        """

        self.population[0] = Indiv(objfunc, objfunc.random_solution())

    
    def perturb(self, indiv_list, objfunc, progress=0, history=None):
        """
        Performs a step of the algorithm
        """

        indiv = indiv_list[0]

        # Perturb individual
        new_solution = self.perturb_op(indiv, indiv_list, objfunc)
        new_solution = objfunc.repair_solution(new_solution)
        new_indiv = Indiv(objfunc, new_solution)

        # If it improves the previous solution keep it
        if new_indiv.fitness > indiv.fitness:
            indiv = new_indiv        
                
        return [indiv]

    
    def update_params(self, progress):
        """
        Updates the parameters of each component of the algorithm
        """

        self.perturb_op.step(progress)

    
    
