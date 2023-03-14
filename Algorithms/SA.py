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

        super().__init__(name)

        # Parameters of the algorithm
        self.params = params
        self.iter = params["iter"] if "iter" in params else 100
        self.temp_init = params["temp_init"] if "temp_init" in params else 100
        self.temp = self.temp_init 
        self.alpha = params["alpha"] if "alpha" in params else 0.99

        self.population = [None]
        self.perturb_op = perturb_op
    
    def best_solution(self):
        """
        Gives the best solution found by the algorithm and its fitness
        """

        best_fitness = self.best_indiv.fitness
        if self.best_indiv.objfunc.opt == "min":
            best_fitness *= -1
        return (self.best_indiv.vector, best_fitness)


    def initialize(self, objfunc):
        """
        Generates a random vector as a starting point for the algorithm
        """

        self.population[0] = Indiv(objfunc, objfunc.random_solution())
        self.best_indiv = self.population[0]

    def perturb(self, indiv_list, objfunc, progress=None, history=None):
        """
        Applies a mutation operator to the current individual
        """

        indiv = indiv_list[0]
        for j in range(self.iter):
            new_indiv = self.perturb_op(indiv, indiv_list, objfunc, self.best_indiv)
            new_indiv.vector = objfunc.repair_solution(new_indiv.vector)

            # Store best vector for individual
            new_indiv.store_best(indiv)
            
            # Accept the new solution even if it is worse with a probability
            p = np.exp(-1/self.temp)
            if new_indiv.fitness > indiv.fitness or random.random() < p:
                indiv = new_indiv
            
            if new_indiv.fitness > self.best_indiv.fitness:
                self.best_indiv = new_indiv
            
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
    