import random
import numpy as np

from ..Individual import *
from ...ParamScheduler import ParamScheduler


class SimAnnEvolve:
    """
    Class implementing the Simulated annealing algorithm
    """

    def __init__(self, objfunc, perturb_op, params):
        """
        Constructor of the SimAnnEvolve class
        """

        self.params = params

        # Parameters of the algorithm
        self.iter = params["iter"] if "iter" in params else 100
        self.temp_init = params["temp_init"] if "temp_init" in params else 100
        self.temp = self.temp_init 
        self.alpha = params["alpha"] if "alpha" in params else 0.99
        self.beta = params["beta"] if "beta" in params else 0.1
        self.method = params["sa_method"] if "sa_method" in params else "geo"

        self.best_fit = 0       

        self.current_indiv = None
        self.best_indiv = None
        self.objfunc = objfunc
        self.perturb_op = perturb_op


    def step(self, progress):
        """
        Updates the parameters and the operators
        """

        self.perturb_op.step(progress)

        if isinstance(self.params, ParamScheduler):
            self.params.step(progress)
            self.iter = round(self.params["iter"])
            # self.temp_changes = params["temp_ch"]
            # self.alpha = params["alpha"]
            # self.temp_init = params["temp_init"]
        
        if self.method == "linear":
            self.temp = self.temp - self.beta
        elif self.method == "geo":
            self.temp = self.temp*self.alpha        

        self.temp = max(self.temp, 1e-10)

        


    
    def best_solution(self):
        """
        Gives the best solution found by the algorithm and its fitness
        """

        best_fitness = self.best_indiv.fitness
        if self.objfunc.opt == "min":
            best_fitness *= -1
        return (self.best_indiv.vector, best_fitness)

    def current_solution(self):
        """
        Gives the current solution of the algorithm and its fitness
        """

        curr_fitness = self.current_indiv.fitness
        if self.objfunc.opt == "min":
            curr_fitness *= -1
        return (self.current_indiv.vector, curr_fitness)


    def generate_random(self):
        """
        Generates a random vector as a starting point for the algorithm
        """

        self.current_indiv = Indiv(self.objfunc, self.objfunc.random_solution())
        self.best_indiv = self.current_indiv
        self.best_fit = self.current_indiv.fitness

    def perturb_and_test(self):
        """
        Applies a mutation operator to the current individual
        """

        for j in range(self.iter):
            new_solution = self.perturb_op.evolve(self.current_indiv, [self.current_indiv], self.objfunc)
            # diff = np.abs(new_solution - self.current_indiv.vector)
            new_solution = self.objfunc.check_bounds(new_solution)
            new_indiv = Indiv(self.objfunc, new_solution)

            p = np.exp(-1/self.temp)
            if new_indiv.fitness > self.current_indiv.fitness or random.random() < p:
                self.current_indiv = new_indiv
            
            if new_indiv.fitness > self.best_indiv.fitness:
                self.best_indiv = new_indiv
    