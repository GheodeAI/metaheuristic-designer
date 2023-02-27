import random
import numpy as np
from numba import jit

from ..Individual import *


class InHSPopulation:
    """
    Population of the In-HS algorithm
    """

    def __init__(self, objfunc, mutation_ops, replace_op, params, population=None):
        """
        Constructor of the InHSPopulation class
        """

        # Hyperparameters of the algorithm
        self.size = params["popSize"]
        self.hmcr = params["HMCR"]
        self.par = params["PAR"]
        self.bn = params["BN"]
        self.mutation_ops = mutation_ops
        self.replace_op = replace_op
        self.mutation_levels = len(mutation_op)

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

        for op in self.mutation_ops:
            op.step(progress)
        
        self.replace_op.step(progress)
        
        if isinstance(self.params, ParamScheduler):
            self.params.step(progress)
            self.size = params["popSize"]
            self.hmcr = params["HMCR"]
            self.par = params["PAR"]
            self.bn = params["BN"]

    
    def best_solution(self):
        """
        Gives the best solution found by the algorithm and its fitness
        """

        best_solution = sorted(self.population, reverse=True, key = lambda c: c.get_fitness())[0]
        best_fitness = best_solution.get_fitness()
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
        Applies the HS operation to the population
        """

        # popul_matrix = HM
        popul_matrix = np.vstack([i.vector for i in self.population])
        fitness_values = np.array([ind.get_fitness() for ind in self.population])
        popul_matrix = popul_matrix[np.argsort(fitness_values)]
        popul_matrix = np.flip(popul_matrix, axis=0)
        solution_size = popul_matrix.shape[1]
        new_solution = np.zeros(solution_size)
        mask1 = np.zeros(solution_size)
        mask2 = np.ones(solution_size)
        mask3 = -np.ones(solution_size)
        for i in range(solution_size):
            if random.random() < self.hmcr:
                pos = random.randrange(popul_matrix.shape[0])
                new_solution[i] = popul_matrix[pos][i]
                mask2[i] = 0
                mask3[i] = pos
                if random.random() <= self.par:
                    mask1[i] = 1
        mask1 = mask1 >=1
        mask2 = mask2 >=1
        mutation = np.zeros(solution_size)
        for i in range(self.mutation_levels):
            step = solution_size/self.mutation_levels
            mask1_aux = (mask3 >= int(step*i)) & (mask3 < int(step*i + step)) & mask1
            mutation_i = self.mutation_ops[i].evolve(Indiv(self.objfunc, new_solution), self.population, self.objfunc)
            new_solution[mask1_aux] = mutation_i[mask1_aux]
            mutation[int(step*i):int(step*i+step)] = mutation_i[int(step*i):int(step*i+step)]
            new_solution[mask2] = self.replace_op.evolve(Indiv(self.objfunc, new_solution), self.population, self.objfunc)[mask2]
        
        mask3[mask1==False] = -1
        popul_matrix = self.inharmony(mask3, popul_matrix, mutation)

        self.population += [Indiv(self.objfunc, sol) for sol in popul_matrix]

        new_solution = self.objfunc.check_bounds(new_solution)
        self.population.append(Indiv(self.objfunc, new_solution))

    def inharmony(self, positions, ordered_population, mutation):
        """
        """
        
        tam = len(ordered_population)
        new_population = []
        for i in range(len(positions)):
            pos = int(positions[i])
            if pos != -1:
                if pos==0:
                    ind = ordered_population[pos+1]
                    inharmony_mutation = mutation[i]/(tam-pos)
                    ind[i] = ind[i]+inharmony_mutation
                    ind = self.objfunc.check_bounds(ind)
                    new_population.append(ind)
                elif pos==(len(ordered_population)-1):
                    ind = ordered_population[pos-1]
                    inharmony_mutation = mutation[i]/(tam-pos)
                    ind[i] = ind[i]+inharmony_mutation
                    ind = self.objfunc.check_bounds(ind)
                    new_population.append(ind)
                else:
                    ind_pre = ordered_population[pos-1]
                    ind_post = ordered_population[pos+1]
                    inharmony_mutation = mutation[i]/(tam-pos)
                    ind_pre[i] = ind_pre[i]+inharmony_mutation
                    ind_post[i] = ind_post[i]+inharmony_mutation
                    ind_pre = self.objfunc.check_bounds(ind_pre)
                    ind_post = self.objfunc.check_bounds(ind_post)
                    new_population.append(ind_pre)
                    new_population.append(ind_post)
        return new_population
    

    def selection(self):
        """
        Selects the individuals that will pass to the next generation
        """

        actual_size_pop = len(self.population)
        fitness_values = np.array([ind.get_fitness() for ind in self.population])
        #print(fitness_values)
        kept_ind = list(np.argsort(fitness_values))[:(actual_size_pop - self.size)]

        self.population = [self.population[i] for i in range(len(self.population)) if i not in kept_ind] 

