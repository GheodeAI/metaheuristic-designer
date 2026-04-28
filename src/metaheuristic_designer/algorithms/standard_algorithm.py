from __future__ import annotations
import logging
from copy import copy
from ..algorithm import Algorithm

logger = logging.getLogger(__name__)


class StandardAlgorithm(Algorithm):
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

    def _log_debug(self, text, population):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(text, population.debug_repr())

    def step(self, population=None):
        # Get the population of this generation
        if population is None:
            population = self.search_strategy.population
        else:
            self.search_strategy.population = population

        self._log_debug("Original population:\n%s", population)

        # Generate their parents
        parents = self.search_strategy.select_parents(population)
        self._log_debug("Parent selection\n%s", parents)

        # Evolve the selected parents
        offspring = self.search_strategy.perturb(parents)
        self._log_debug("Perturbed\n%s", offspring)

        # Get the fitness of the individuals
        offspring = self.search_strategy.evaluate_population(offspring, self.parallel, self.threads)
        self._log_debug("Evaluated\n%s", offspring)

        # Select the individuals that remain for the next generation
        new_population = self.search_strategy.select_individuals(population, offspring)
        self._log_debug("Selected\n%s", new_population)

        self.search_strategy.population = new_population

        # Get information about the algorithm to track it's progress
        self.search_strategy.step(progress=self.progress)
        self._log_debug("Updated end\n%s", new_population)

        # Store information
        best_individual, best_fitness = self.search_strategy.best_solution()
        self.best_history.append(best_individual)
        self.fit_history.append(best_fitness)

        return new_population
