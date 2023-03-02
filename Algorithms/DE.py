import random
import numpy as np
from ..Individual import Indiv
from ..ParamScheduler import ParamScheduler
from ..SurvivorSelection import SurvivorSelection
from .BaseAlgorithm import BaseAlgorithm


class DE(BaseAlgorithm):
    """
    Population of the Genetic algorithm
    """

    def __init__(self, objfunc, de_op, params={}, selection_op=None, name="DE", population=None):
        """
        Constructor of the GeneticPopulation class
        """

        super().__init__(objfunc, name)

        # Hyperparameters of the algorithm
        self.params = params
        self.size = params["popSize"] if "popSize" in params else 100
        self.de_op = de_op

        if selection_op is None:
            selection_op = SurvivorSelection("One-to-one")
        self.selection_op = selection_op
        

        # Population initialization
        if population is not None:
            self.population = population

    def best_solution(self):
        """
        Gives the best solution found by the algorithm and its fitness
        """

        best_solution = sorted(self.population, reverse=True, key = lambda c: c.fitness)[0]
        best_fitness = best_solution.fitness
        if self.objfunc.opt == "min":
            best_fitness *= -1
        return (best_solution.vector, best_fitness)

    def initialize(self):
        """
        Generates a random population of individuals
        """

        self.population = []
        for i in range(self.size):
            new_ind = Indiv(self.objfunc, self.objfunc.random_solution())
            self.population.append(new_ind)
    
    def perturb(self, parent_list, progress=0, history=None):
        offspring = []

        for indiv in parent_list:
            new_solution = self.de_op(indiv, self.population, self.objfunc)
            new_solution = self.objfunc.check_bounds(new_solution)
            new_ind = Indiv(self.objfunc, new_solution)
            
            offspring.append(new_ind)
        
        return offspring
    
    def select_individuals(self, population, offspring, progress=0, history=None):
        return self.selection_op(population, offspring)

    def update_params(self, progress):
        """
        Updates the parameters and the operators
        """

        self.de_op.step(progress)
        self.selection_op.step(progress)

        if isinstance(self.params, ParamScheduler):
            self.params.step(progress)
            self.size = self.params["popSize"]





