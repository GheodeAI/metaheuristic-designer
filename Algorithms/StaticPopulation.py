import random
import numpy as np
from typing import List, Union
from ..Individual import Indiv
from ..ParamScheduler import ParamScheduler
from ..Operators import Operator
from ..SurvivorSelection import SurvivorSelection
from .BaseAlgorithm import BaseAlgorithm


class StaticPopulation(BaseAlgorithm):
    """
    Population of the Genetic algorithm
    """

    def __init__(self, operator: Operator, params: Union[ParamScheduler, dict]={}, selection_op: SurvivorSelection=None, name: str="stpop", population: List[Indiv]=None):
        """
        Constructor of the GeneticPopulation class
        """

        super().__init__(name)

        # Hyperparameters of the algorithm
        self.params = params
        self.size = params["popSize"] if "popSize" in params else 100
        self.operator = operator

        if selection_op is None:
            selection_op = SurvivorSelection("Generational")
        self.selection_op = selection_op

        self.best = None

        # Population initialization
        if population is not None:
            self.population = population

    def best_solution(self):
        """
        Gives the best solution found by the algorithm and its fitness
        """

        best_fitness = self.best.fitness
        if self.best.objfunc.opt == "min":
            best_fitness *= -1        

        return self.best.vector, best_fitness

    def initialize(self, objfunc):
        """
        Generates a random population of individuals
        """

        self.population = []
        for i in range(self.size):
            new_indiv = Indiv(objfunc, objfunc.random_solution())

            if self.best is None or self.best.fitness < new_indiv.fitness:
                self.best = new_indiv
            
            self.population.append(new_indiv)
    
    def perturb(self, parent_list, objfunc, progress=0, history=None):
        offspring = []
        for indiv in parent_list:

            # Apply operator
            new_indiv = self.operator(indiv, parent_list, objfunc, self.best)
            new_indiv.vector = objfunc.repair_solution(new_indiv.vector)
            new_indiv.speed = objfunc.repair_solution(new_indiv.speed)
            
            # Add to offspring list
            offspring.append(new_indiv)
        
        # Update best solution
        current_best = max(offspring, key = lambda x: x.fitness)
        if self.best.fitness < current_best.fitness:
            self.best = current_best
        
        return offspring
    
    def select_individuals(self, population, offspring, progress=0, history=None):
        return self.selection_op(population, offspring)

    def update_params(self, progress):
        """
        Updates the parameters and the operators
        """

        self.operator.step(progress)
        self.selection_op.step(progress)

        if isinstance(self.params, ParamScheduler):
            self.params.step(progress)
            self.size = self.params["popSize"]





