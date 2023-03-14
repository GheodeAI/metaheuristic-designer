import random
import numpy as np
from typing import Union
from ..Individual import Indiv
from ..ParamScheduler import ParamScheduler
from ..Operators import Operator
from ..BaseAlgorithm import BaseAlgorithm



class SA(BaseAlgorithm):
    """
    Class implementing the Simulated annealing algorithm
    """

    def __init__(self, perturb_op: Operator, params: Union[ParamScheduler, dict]={}, name: str="SA"):
        """
        Constructor of the SimAnnEvolve class
        """

        

        # Parameters of the algorithm
        self.params = params
        self.iter = params["iter"] if "iter" in params else 100
        self.temp_init = params["temp_init"] if "temp_init" in params else 100
        self.temp = self.temp_init 
        self.alpha = params["alpha"] if "alpha" in params else 0.99

        self.population = [None]
        self.perturb_op = perturb_op

        super().__init__(name, popSize=1)
    

    def perturb(self, indiv_list, objfunc, progress=None, history=None):
        """
        Applies a mutation operator to the current individual
        """

        indiv = indiv_list[0]
        for j in range(self.iter):
            new_indiv = self.perturb_op(indiv, indiv_list, objfunc, self.best)
            new_indiv.genotype = objfunc.repair_solution(new_indiv.genotype)

            # Store best vector for individual
            new_indiv.store_best(indiv)
            
            # Accept the new solution even if it is worse with a probability
            p = np.exp(-1/self.temp)
            if new_indiv.fitness > indiv.fitness or random.random() < p:
                indiv = new_indiv
            
            if new_indiv.fitness > self.best.fitness:
                self.best = new_indiv
            
        return [indiv]
        
    
    def update_params(self, progress):
        """
        Updates the parameters and the operators
        """

        self.perturb_op.step(progress)

        if isinstance(self.params, ParamScheduler):
            self.params.step(progress)
            self.iter = round(self.params["iter"])
            self.alpha = params["alpha"]
        
        self.temp = self.temp*self.alpha
    

    def extra_step_info(self):
        """
        Specific information to display relevant to this algorithm
        """

        print(f"\ttemperature: {float(self.temp):0.3}")
        print(f"\taccept prob: {np.exp(-1/self.temp):0.3}")
    