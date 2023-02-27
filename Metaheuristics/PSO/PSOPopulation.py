import random
import numpy as np
from numba import jit
import time

from ..Individual import *
from ...ParamScheduler import ParamScheduler


class PSOPopulation:    
    """
    Population of the PSO algorithm
    """

    def __init__(self, objfunc, params, population=None):
        """
        Constructor of the PSOPopulation class
        """

        self.params = params

        # Hyperparameters of the algorithm
        self.size = params["popSize"] if "popSize" in params else 100
        self.w = params["w"] if "w" in params else 0.7
        self.c1 = params["c1"] if "c1" in params else 1.5
        self.c2 = params["c2"] if "c2" in params else 1.5

        # Data structures of the algorithm
        self.objfunc = objfunc
        self.global_best = None

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

        if isinstance(self.params, ParamScheduler):
            self.params.step(progress)
            self.size = self.params["popSize"]
            self.w = self.params["w"]
            self.c1 = self.params["c1"]
            self.c2 = self.params["c2"]     


    def best_solution(self):
        """
        Gives the best solution found by the algorithm and its fitness
        """
        
        best_solution = self.global_best
        best_fitness = self.global_best_fit
        if self.objfunc.opt == "min":
            best_fitness *= -1
        return (best_solution, best_fitness)

    
    def generate_random(self):
        """
        Generates a random population of individuals
        """

        self.population = []
        for i in range(self.size):
            new_ind = Indiv(self.objfunc, self.objfunc.random_solution(), self.objfunc.random_solution()*0.05)
            self.population.append(new_ind)
    
    
    def particle_step(self):
        """
        Applies the particle swarm operator to the population of individuals 
        """

        fitness_list = [i.fitness for i in self.population]
        best_idx = fitness_list.index(max(fitness_list))
        best_particle = self.population[best_idx]
        
        if self.global_best is None:
            self.global_best = best_particle.vector
            self.global_best_fit = best_particle.fitness

        for idx, val in enumerate(self.population):
            r1, r2 = np.random.random(val.vector.shape), np.random.random(val.vector.shape) 
            
            new_speed = self.w * val.speed + self.c1 * r1 * (val.best - val.vector) + self.c2 * r2 * (self.global_best - val.vector)
            val.speed = new_speed
            new_particle = val.apply_speed()
            self.population[idx] = new_particle

            if new_particle.fitness >= self.global_best_fit:
                self.global_best = new_particle.vector
                self.global_best_fit = new_particle.fitness  



