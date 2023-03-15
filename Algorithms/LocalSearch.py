import random
import numpy as np
from typing import Union
from ..Individual import Indiv
from ..ParamScheduler import ParamScheduler
from ..Operators import Operator
from ..BaseAlgorithm import BaseAlgorithm

class LocalSearch(BaseAlgorithm):
    """
    Search strtategy example, HillClimbing
    """
    
    def __init__(self, perturb_op: Operator, params: Union[ParamScheduler, dict]={}, name: str="LocalSearch"):
        """
        Constructor of the Example search strategy class
        """

        self.perturb_op = perturb_op
        self.iterations = params["iters"] if "iters" in params else 100

        super().__init__(name, popSize=1)


    def perturb(self, indiv_list, objfunc, progress=0, history=None):
        result = []

        for indiv in indiv_list:
            best_indiv = indiv
            
            for i in range(self.iterations):
    
                # Perturb individual
                new_indiv = self.perturb_op(indiv, indiv_list, objfunc, self.best)
                new_indiv.genotype = objfunc.repair_solution(new_indiv.genotype)

                # Store best vector for individual
                new_indiv.store_best(indiv)

                # If it improves the best solution keep it
                if new_indiv.fitness > best_indiv.fitness:
                    best_indiv = new_indiv
                
            result.append(best_indiv)
            
        curr_best = max(result, key=lambda x: x.fitness)
        if curr_best.fitness > self.best.fitness:
            self.best = curr_best
        
        return result

    
    def update_params(self, progress):
        """
        Updates the parameters of each component of the algorithm
        """

        self.perturb_op.step(progress)

    
    
