from __future__ import annotations
import time
from matplotlib import pyplot as plt
from ..Algorithm import Algorithm
from ..ObjectiveFunc import ObjectiveFunc
from ..SearchStrategy import SearchStrategy
from ..ParamScheduler import ParamScheduler


class GeneralAlgorithm(Algorithm):
    """
    General framework for metaheuristic algorithms.

    Performs a loop of parent selection, perturbation, and survivor selection until a stopping condition is reached.

    Parameters
    ----------

    objfunc: ObjectiveFunc
        Objective function to be optimized.
    search_strategy: Algorithm
        Search strategy that will iteratively optimize the function.
    params: ParamScheduler or dict, optional
        Dictionary of parameters to define the stopping condition and output of the algorithm.
    """

    def step(self, time_start=0, verbose=False):
        # Get the population of this generation
        population = self.search_strategy.population
        # print()
        # print("Previous population: ", population)

        # Generate their parents
        parents = self.search_strategy.select_parents(population, progress=self.progress, history=self.best_history)
        # print()
        # print("Parents: ", parents)

        # Evolve the selected parents
        offspring = self.search_strategy.perturb(parents, progress=self.progress, history=self.best_history)
        # print()
        # print("Offspring: ", parents)

        # Get the fitness of the individuals
        offspring = self.search_strategy.evaluate_population(offspring, self.parallel, self.threads)
        # print()
        # print("Evaluated offspring: ", parents)

        # Select the individuals that remain for the next generation
        population = self.search_strategy.select_individuals(population, offspring, progress=self.progress, history=self.best_history)
        # print()
        # print("Selected offspring: ", parents)

        # Assign the newly generate population
        self.search_strategy.population = population

        # Get information about the algorithm to track it's progress
        self.search_strategy.update_params(progress=self.progress)

        # Store information
        best_individual, best_fitness = self.search_strategy.best_solution()
        self.best_history.append(best_individual)
        self.fit_history.append(best_fitness)

        return population
