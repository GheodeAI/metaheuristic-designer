import random
import numpy as np
from ..Individual import Indiv
from ..ParamScheduler import ParamScheduler
from ..Operators import Operator
from ..BaseAlgorithm import BaseAlgorithm


class HillClimb(BaseAlgorithm):
    """
    Search strtategy example, HillClimbing
    """
    
    def __init__(self, perturb_op: Operator, name: str="HillClimb"):
        """
        Constructor of the Example search strategy class
        """

        self.perturb_op = perturb_op

        super().__init__(name, popSize=1)

    
    def perturb(self, indiv_list, objfunc, progress=0, history=None):
        """
        Performs a step of the algorithm
        """

        indiv = indiv_list[0]

        # Perturb individual
        new_indiv = self.perturb_op(indiv, indiv_list, objfunc, self.best)
        new_indiv.vector = objfunc.repair_solution(new_indiv.vector)
    
        # Store best vector for individual
        new_indiv.store_best(indiv)

        # If it improves the previous solution keep it
        if new_indiv.fitness > indiv.fitness:
            indiv = new_indiv       

        if new_indiv.fitness > self.best.fitness:
            self.best = new_indiv
                
        return [indiv]

    
    def update_params(self, progress):
        """
        Updates the parameters of each component of the algorithm
        """

        self.perturb_op.step(progress)

    
    
