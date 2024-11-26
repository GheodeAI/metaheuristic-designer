from __future__ import annotations
from ..Algorithm import Algorithm

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
        population = population.sort_population()

        # Generate their parents
        parents = self.search_strategy.select_parents(population, progress=self.progress, history=self.best_history)

        # Evolve the selected parents
        offspring = self.search_strategy.perturb(parents, progress=self.progress, history=self.best_history)

        # Get the fitness of the individuals
        offspring = self.search_strategy.evaluate_population(offspring, self.parallel, self.threads)

        # Select the individuals that remain for the next generation
        new_population = self.search_strategy.select_individuals(population, offspring, progress=self.progress, history=self.best_history)

        # Assign the newly generated population
        self.search_strategy.population = new_population
        
        # Get information about the algorithm to track it's progress
        self.search_strategy.update_params(progress=self.progress)

        # Store information
        best_individual, best_fitness = self.search_strategy.best_solution()
        self.best_history.append(best_individual)
        self.fit_history.append(best_fitness)

        return new_population
