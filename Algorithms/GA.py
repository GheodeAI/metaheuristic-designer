import random
import numpy as np
from copy import copy
import time
from ..Individual import Indiv
from ..ParamScheduler import ParamScheduler
from .BaseAlgorithm import BaseAlgorithm


class GA(BaseAlgorithm):
    """
    Population of the Genetic algorithm
    """

    def __init__(self, mutation_op, cross_op, parent_sel_op, selection_op, objfunc=None, params={}, name="GA", population=None):
        """
        Constructor of the GeneticPopulation class
        """

        super().__init__(name)

        # Hyperparameters of the algorithm
        self.params = params
        self.size = params["popSize"] if "popSize" in params else 100
        self.pmut = params["pmut"] if "pmut" in params else 0.1
        self.pcross = params["pcross"] if "pcross" in params else 0.9
        self.mutation_op = mutation_op
        self.cross_op = cross_op
        self.parent_sel_op = parent_sel_op
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
            new_ind = Indiv(objfunc.random_solution())
            self.population.append(new_ind)
    

    def select_parents(self, population, progress=0, history=None):
        return self.parent_sel_op(population)
    
    def perturb(self, parent_list, objfunc, progress=0, history=None):
        # Generation of offspring by crossing and mutation
        offspring = []
        while len(offspring) < self.size:

            # Cross
            parent1 = random.choice(parent_list)
            if random.random() < self.pcross:
                new_solution = self.cross_op.evolve(parent1, parent_list, objfunc)
                new_solution = objfunc.repair_solution(new_solution)
                new_ind = Indiv(new_solution)
            else:
                new_ind = copy(parent1)
            
            # Mutate
            if random.random() < self.pmut:
                new_solution = self.mutation_op(new_ind, self.population, objfunc)
                new_solution = objfunc.repair_solution(new_solution)
                new_ind = Indiv(new_solution)
            
            # Add to offspring list
            offspring.append(new_ind)
        
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


