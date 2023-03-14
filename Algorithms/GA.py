import random
import numpy as np
from copy import copy
import time
from typing import Union, List
from ..Individual import Indiv
from ..Operators import Operator
from ..ParentSelection import ParentSelection
from ..SurvivorSelection import SurvivorSelection
from ..ParamScheduler import ParamScheduler
from ..BaseAlgorithm import BaseAlgorithm


class GA(BaseAlgorithm):
    """
    Population of the Genetic algorithm
    """

    def __init__(self, mutation_op: Operator, cross_op: Operator, parent_sel_op: ParentSelection, selection_op: SurvivorSelection,
                       params: Union[ParamScheduler, dict]={}, name: str="GA", population: List[Indiv]=None):
        """
        Constructor of the GeneticPopulation class
        """

        # Hyperparameters of the algorithm
        self.params = params
        self.popsize = params["popSize"] if "popSize" in params else 100
        self.pmut = params["pmut"] if "pmut" in params else 0.1
        self.pcross = params["pcross"] if "pcross" in params else 0.9

        self.mutation_op = mutation_op
        self.cross_op = cross_op

        self.parent_sel_op = parent_sel_op
        self.selection_op = selection_op

        self.best = None

        # Population initialization
        if population is not None:
            self.population = population
        
        super().__init__(name, self.popsize)


    def select_parents(self, population, progress=0, history=None):
        return self.parent_sel_op(population)
    

    def perturb(self, parent_list, objfunc, progress=0, history=None):
        # Generation of offspring by crossing and mutation
        offspring = []
        while len(offspring) < self.popsize:

            # Cross
            parent1 = random.choice(parent_list)
            if random.random() < self.pcross:
                new_indiv = self.cross_op(parent1, parent_list, objfunc, self.best)
                new_indiv.vector = objfunc.repair_solution(new_indiv.vector)
            else:
                new_indiv = copy(parent1)
            
            # Mutate
            if random.random() < self.pmut:
                new_indiv = self.mutation_op(parent1, parent_list, objfunc, self.best)
                new_indiv.vector = objfunc.repair_solution(new_indiv.vector)
            
            # Store best vector for individual
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
            self.pmut = self.params["pmut"]
            self.pcross = self.params["pcross"]
    
    def extra_step_info(self):
        """
        Specific information to display relevant to this algorithm
        """
        popul_matrix = np.array(list(map(lambda x: x.vector, self.population)))
        divesity = popul_matrix.std(axis=1).mean()
        print(f"\tdiversity: {divesity:0.3}")


