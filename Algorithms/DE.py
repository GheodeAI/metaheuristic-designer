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

    def __init__(self, de_op, params={}, selection_op=None, name="DE", population=None):
        """
        Constructor of the GeneticPopulation class
        """

        super().__init__(name)

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

    def best_solution(self, objfunc):
        """
        Gives the best solution found by the algorithm and its fitness
        """

        best_solution = sorted(self.population, reverse=True, key = lambda c: c.fitness)[0]
        best_fitness = best_solution.fitness
        if objfunc.opt == "min":
            best_fitness *= -1
        return (best_solution.vector, best_fitness)

    def initialize(self, objfunc):
        """
        Generates a random population of individuals
        """

        self.population = []
        for i in range(self.size):
            new_indiv = Indiv(objfunc, objfunc.random_solution())
            self.population.append(new_indiv)
    
    def perturb(self, parent_list, objfunc, progress=0, history=None):
        offspring = []

        for indiv in parent_list:
            new_solution = self.de_op(indiv, parent_list, objfunc)
            new_solution = objfunc.repair_solution(new_solution)
            new_indiv = Indiv(objfunc, new_solution)
            
            offspring.append(new_indiv)
        
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





