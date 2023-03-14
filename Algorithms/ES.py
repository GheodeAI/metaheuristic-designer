import random
import numpy as np
import time
from typing import Union, List
from ..Individual import Indiv
from ..ParamScheduler import ParamScheduler
from ..Operators import Operator
from ..ParentSelection import ParentSelection
from ..SurvivorSelection import SurvivorSelection
from ..BaseAlgorithm import BaseAlgorithm


class ES(BaseAlgorithm):
    """
    Population of the Genetic algorithm
    """

    def __init__(self, mutation_op: Operator, cross_op: Operator, parent_sel_op: ParentSelection, selection_op: SurvivorSelection, 
                       params: Union[ParamScheduler, dict] = {}, name: str = "ES", population: List[Indiv] = None):
        """
        Constructor of the GeneticPopulation class
        """

        # Hyperparameters of the algorithm
        self.params = params
        self.popsize = params["popSize"] if "popSize" in params else 100
        self.n_offspring = params["offspringSize"] if "offspringSize" in params else self.size
        self.mutation_op = mutation_op
        self.cross_op = cross_op
        self.parent_sel_op = parent_sel_op
        self.selection_op = selection_op

        # Population initialization
        if population is not None:
            self.population = population
        
        super().__init__(name, self.popsize)
    
    def select_parents(self, population, progress=0, history=None):
        return self.parent_sel_op(population)


    def perturb(self, parent_list, objfunc, progress=0, history=None):
        # Generation of offspring by crossing and mutation
        offspring = []

        while len(offspring) < self.n_offspring:

            # Cross
            parent1 = random.choice(parent_list)
            new_indiv = self.cross_op(parent1, parent_list, objfunc, self.best)
            new_indiv.vector = objfunc.repair_solution(new_indiv.vector)

            # Mutate
            new_indiv = self.mutation_op(parent1, parent_list, objfunc, self.best)
            new_indiv.vector = objfunc.repair_solution(new_indiv.vector)

            # Store best vector for individual (useful for some operators, not extrictly needed)
            new_indiv.store_best(parent1)

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

        self.mutation_op.step(progress)
        self.cross_op.step(progress)
        self.parent_sel_op.step(progress)
        self.selection_op.step(progress)

        if isinstance(self.params, ParamScheduler):
            self.params.step(progress)
            self.size = self.params["popSize"]
            self.n_offspring = params["offspringSize"]




