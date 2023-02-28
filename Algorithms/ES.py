import random
import numpy as np
from numba import jit
from copy import deepcopy
from ..Individual import *
from ..ParamScheduler import ParamScheduler


class ES:
    """
    Population of the Genetic algorithm
    """

    def __init__(self, objfunc, mutation_op, cross_op, parent_sel_op, selection_op, params, name="ES", population=None):
        """
        Constructor of the GeneticPopulation class
        """

        self.name = name
        self.params = params

        # Hyperparameters of the algorithm
        self.size = params["popSize"] if "popSize" in params else 100
        self.n_offspring = params["offspringSize"] if "offspringSize" in params else self.size
        self.mutation_op = mutation_op
        self.cross_op = cross_op
        self.parent_sel_op = parent_sel_op
        self.selection_op = selection_op

        # Data structures of the algorithm
        self.objfunc = objfunc

        # Population initialization
        if population is None:
            self.population = []
        else:
            self.population = population
        self.offspring = []       

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
    

    def step(self, progress, history=None):
        """
        Performs a step of the algorithm
        """

        # Parent selection
        parent_list = self.parent_sel_op(self.population)

        # Generation of offspring by crossing and mutation
        self.offspring = []
        while len(self.offspring) < self.n_offspring:

            # Cross
            parent1 = random.choice(parent_list)
            new_solution = self.cross_op.evolve(parent1, parent_list, self.objfunc)
            new_solution = self.objfunc.check_bounds(new_solution)
            new_ind = Indiv(self.objfunc, new_solution)
            
            # Mutate
            new_solution = self.mutation_op(new_ind, self.population, self.objfunc)
            new_solution = self.objfunc.check_bounds(new_solution)
            new_ind = Indiv(self.objfunc, new_solution)
            
            # Add to offspring list
            self.offspring.append(new_ind)

        self.population = self.selection_op(self.population, self.offspring)
        return self.best_solution()

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




