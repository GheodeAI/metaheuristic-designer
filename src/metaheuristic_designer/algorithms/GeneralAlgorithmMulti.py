from __future__ import annotations
import time
import warnings
from matplotlib import pyplot as plt
from ..Algorithm import Algorithm, process_condition


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

    def __init__(
        self,
        objfunc: ObjectiveFunc,
        search_strategy: SearchStrategy,
        params: Union[ParamScheduler, dict] = None,
        name: str = None,
    ):
        """
        Constructor of the Metaheuristic class
        """
        super().__init__(objfunc, search_strategy, params, name)

        if self.progress_metric_used["fit_target"] or self.stop_cond_used["fit_target"]:
            warnings.warn("Target fitness not available as a stopping condition for multiobjective optimization.")
            self.stop_cond_used["fit_target"] = False
            self.progress_metric_used["fit_target"] = False

        if self.progress_metric_used["convergence"] or self.stop_cond_used["convergence"]:
            warnings.warn("Convergence not available as a stopping condition for multiobjective optimization.")
            self.stop_cond_used["convergence"] = False
            self.progress_metric_used["convergence"] = False

    
    def result(self):
        return self.search_strategy.population
    
    def step(self, time_start=0, verbose=False):
        # Get the population of this generation
        population = self.search_strategy.population

        # Generate their parents
        parents = self.search_strategy.select_parents(population, progress=self.progress, history=self.best_history)

        # Evolve the selected parents
        offspring = self.search_strategy.perturb(parents, self.objfunc, progress=self.progress, history=self.best_history)

        # Get the fitness of the individuals
        offspring = self.search_strategy.evaluate_population(offspring, self.objfunc, self.parallel, self.threads)

        # Select the individuals that remain for the next generation
        population = self.search_strategy.select_individuals(population, offspring, progress=self.progress, history=self.best_history)

        # Assign the newly generate population
        self.search_strategy.population = population

        # Get information about the algorithm to track it's progress
        best_individual, best_fitness = self.search_strategy.best_solution()
        self.search_strategy.update_params(progress=self.progress)

        # Store information
        # self.best_history.append(best_individual)
        # self.fit_history.append(best_fitness)

        return (best_individual, best_fitness)

    def update(self, real_time_start, cpu_time_start, pass_step=True):
        if pass_step:
            self.steps += 1

        self.progress = self.get_progress(self.steps, real_time_start, cpu_time_start)

        self.ended = self.stopping_condition(self.steps, real_time_start, cpu_time_start)

