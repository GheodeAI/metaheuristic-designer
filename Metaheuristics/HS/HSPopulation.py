import random
import numpy as np
from numba import jit

from ..Individual import *
from ...ParamScheduler import ParamScheduler


class HSPopulation:
    """
    Population of the HS algorithm
    """

    def __init__(self, objfunc, mutation_op, replace_op, params, population=None):
        """
        Constructor of the HSPopulation class
        """
        self.params = params

        # Hyperparameters of the algorithm
        self.size = params["popSize"] if "popSize" in params else 100
        self.hmcr = params["HMCR"] if "HMCR" in params else 0.9
        self.par = params["PAR"] if "PAR" in params else 0.3
        self.bn = params["BN"] if "BN" in params else 1
        self.mutation_op = mutation_op
        self.replace_op = replace_op

        # Data structures of the algorithm
        self.objfunc = objfunc

        # Population initialization
        if population is None:
            self.population = []
        else:
            self.population = population
        self.offspring = []       


    def step(self, progress):
        """
        Updates the parameters and the operators
        """

        self.mutation_op.step(progress)
        self.replace_op.step(progress)

        if isinstance(self.params, ParamScheduler):
            self.params.step(progress)
            self.size = self.params["popSize"]
            self.hmcr = self.params["HMCR"]
            self.par = self.params["PAR"]
            self.bn = self.params["BN"]


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
        Applies Harmony search operators to the population of individuals
        """

        # popul_matrix = HM
        popul_matrix = np.vstack([i.vector for i in self.population])
        new_solution = np.zeros(popul_matrix.shape[1])
        mask1 = np.zeros(popul_matrix.shape[1])
        mask2 = np.ones(popul_matrix.shape[1])
        popul_mean = popul_matrix.mean(axis=0)
        for i in range(new_solution.size):
            if random.random() < self.hmcr:
                new_solution[i] = random.choice(self.population).vector[i]
                mask2[i] = 0
                if random.random() <= self.par:
                    mask1[i] = 1
        mask1 = mask1 >=1
        mask2 = mask2 >=1
        new_solution[mask1] = self.mutation_op.evolve(Indiv(self.objfunc, new_solution), self.population, self.objfunc)[mask1]
        new_solution[mask2] = self.replace_op.evolve(Indiv(self.objfunc, new_solution), self.population, self.objfunc)[mask2]
        new_solution = self.objfunc.check_bounds(new_solution)
        self.population.append(Indiv(self.objfunc, new_solution))
    
    
    def selection(self):
        """
        Selects the individuals that will pass to the next generation
        """

        actual_size_pop = len(self.population)
        fitness_values = np.array([ind.fitness for ind in self.population])
        kept_ind = list(np.argsort(fitness_values))[:(actual_size_pop - self.size)]

        self.population = [self.population[i] for i in range(len(self.population)) if i not in kept_ind] 

