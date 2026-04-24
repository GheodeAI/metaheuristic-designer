from __future__ import annotations
import logging
from copy import copy
from ..algorithm import Algorithm

logger = logging.getLogger(__name__)

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

    def step(self, population=None, time_start=0, verbose=False):
        # Get the population of this generation
        if population is None:
            population = self.search_strategy.population
        else:
            self.search_strategy.population = population

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Original population:\n%s", population.debug_repr())

        new_population = copy(population)

        # Generate their parents
        parents = self.search_strategy.select_parents(new_population, progress=self.progress, history=self.best_history)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Parent selection\n%s", parents.debug_repr())

        # Evolve the selected parents
        offspring = self.search_strategy.perturb(parents, progress=self.progress, history=self.best_history)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Perturbed\n%s", offspring.debug_repr())

        # Get the fitness of the individuals
        offspring = self.search_strategy.evaluate_population(offspring, self.parallel, self.threads)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Evaluated\n%s", offspring.debug_repr())

        # Select the individuals that remain for the next generation
        new_population = self.search_strategy.select_individuals(population, offspring, progress=self.progress, history=self.best_history)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Selected\n%s", new_population.debug_repr())

        self.search_strategy.population = new_population

        # Get information about the algorithm to track it's progress
        self.search_strategy.update_params(progress=self.progress)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Updated end\n%s", new_population.debug_repr())

        # Store information
        best_individual, best_fitness = self.search_strategy.best_solution()
        self.best_history.append(best_individual)
        self.fit_history.append(best_fitness)

        return new_population
