from __future__ import annotations
import random
import numpy as np
from typing import List, Union
from ..Individual import Individual
from ..ParamScheduler import ParamScheduler
from ..SurvivorSelection import SurvivorSelection
from ..Algorithm import Algorithm


class StaticPopulation(Algorithm):
    """
    Population of the Genetic algorithm
    """

    def __init__(self, operator: Operator, params: Union[ParamScheduler, dict]={}, selection_op: SurvivorSelection=None, name: str="stpop", population: List[Individual]=None):
        """
        Constructor of the GeneticPopulation class
        """

        # Hyperparameters of the algorithm
        self.params = params
        self.operator = operator

        if selection_op is None:
            selection_op = SurvivorSelection("Generational")
        self.selection_op = selection_op

        self.best = None

        # Population initialization
        if population is not None:
            self.population = population
        
        popsize = params["popSize"] if "popSize" in params else 100
        super().__init__(name, popSize=popsize, params=params)
    
    
    def perturb(self, parent_list, objfunc, progress=0, history=None):
        offspring = []
        for indiv in parent_list:

            # Apply operator
            new_indiv = self.operator(indiv, parent_list, objfunc, self.best)
            new_indiv.genotype = objfunc.repair_solution(new_indiv.genotype)
            new_indiv.speed = objfunc.repair_speed(new_indiv.speed)

            # Store best vector for individual
            new_indiv.store_best(indiv)
            
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

        if self.param_scheduler:
            self.param_scheduler.step(progress)
            self.params = self.param_scheduler.get_params()
            self.size = self.params["popSize"]





