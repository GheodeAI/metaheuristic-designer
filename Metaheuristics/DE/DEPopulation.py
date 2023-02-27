import random
import numpy as np
from numba import jit

from ..Individual import *
from ...ParamScheduler import ParamScheduler


class DEPopulation:    
    """
    Population of the DE algorithm
    """

    def __init__(self, objfunc, diffev_op, replace_op, params, population=None):
        """
        Constructor of the DEPopulation class
        """

        self.params = params

        # Hyperparameters of the algorithm
        self.size = params["popSize"] if "popSize" in params else 100
        self.diffev_op = diffev_op
        self.replace_op = replace_op

        # Data structures of the algorithm
        self.objfunc = objfunc

        # Population initialization
        if population is None:
            self.population = []
        else:
            self.population = population
        self.offspring = [None for i in range(self.size)]
       
    def step(self, progress):
        """
        Updates the parameters and the operators
        """

        self.diffev_op.step(progress)
        self.replace_op.step(progress)

        if isinstance(self.params, ParamScheduler):
            self.params.step(progress)
            self.size = self.params["popSize"]
    

    def best_solution(self):
        """
        Gives the best solution found by the algorithm and its fitness
        """

        best_solution = sorted(self.population, reverse=True, key = lambda c: c.fitness)[0]
        best_fitness = best_solution.fitness
        if self.objfunc.opt == "min":
            best_fitness *= -1
        return (best_solution.vector, best_fitness)


    def generate_random(self):
        """
        Generates a random population of individuals
        """
        
        self.population = []
        for i in range(self.size):
            new_ind = Indiv(self.objfunc, self.objfunc.random_solution())
            self.population.append(new_ind)
            
    
    def evolve(self):
        """
        Applies the DE operator to all the individuals of the population 
        """

        for idx, ind in enumerate(self.population):
            new_solution = self.diffev_op(ind, self.population, self.objfunc)
            new_solution = self.objfunc.check_bounds(new_solution)
            self.offspring[idx] = Indiv(self.objfunc, new_solution)
    
    
    def selection(self):
        """
        Selects the individuals that will pass to the next generation
        """

        self.population = self.replace_op(self.population, self.offspring)

