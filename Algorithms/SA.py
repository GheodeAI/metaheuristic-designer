import random
import numpy as np
from ..Individual import Indiv
from ..ParamScheduler import ParamScheduler
from .BaseAlgorithm import BaseAlgorithm


class SA(BaseAlgorithm):
    """
    Class implementing the Simulated annealing algorithm
    """

    def __init__(self, objfunc, perturb_op, params, name="SA"):
        """
        Constructor of the SimAnnEvolve class
        """

        super().__init__(objfunc, name)

        # Parameters of the algorithm
        self.params = params
        self.iter = params["iter"] if "iter" in params else 100
        self.temp_init = params["temp_init"] if "temp_init" in params else 100
        self.temp = self.temp_init 
        self.alpha = params["alpha"] if "alpha" in params else 0.99

        self.current_indiv = None
        self.best_indiv = None
        self.perturb_op = perturb_op
    
    def best_solution(self):
        """
        Gives the best solution found by the algorithm and its fitness
        """

        best_fitness = self.best_indiv.fitness
        if self.objfunc.opt == "min":
            best_fitness *= -1
        return (self.best_indiv.vector, best_fitness)


    def initialize(self):
        """
        Generates a random vector as a starting point for the algorithm
        """

        self.current_indiv = Indiv(self.objfunc, self.objfunc.random_solution())
        self.best_indiv = self.current_indiv


    def step(self, progress=None, history=None):
        """
        Applies a mutation operator to the current individual
        """

        for j in range(self.iter):
            new_solution = self.perturb_op(self.current_indiv, [self.current_indiv], self.objfunc)
            new_solution = self.objfunc.check_bounds(new_solution)
            new_indiv = Indiv(self.objfunc, new_solution)

            p = np.exp(-1/self.temp)
            if new_indiv.fitness > self.current_indiv.fitness or random.random() < p:
                self.current_indiv = new_indiv
            
            if new_indiv.fitness > self.best_indiv.fitness:
                self.best_indiv = new_indiv
            
        return self.best_solution()
        
    
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

        print(f"\ttemperature: {self.temp:0.3}")
        print(f"\taccept prob: {np.exp(-1/self.temp):0.3}")
    