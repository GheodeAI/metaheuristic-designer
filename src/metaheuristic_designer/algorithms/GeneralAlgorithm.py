from __future__ import annotations
import time
from matplotlib import pyplot as plt
from ..Algorithm import Algorithm


class GeneralAlgorithm(Algorithm):
    """
    General framework for metaheuristic algorithms

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
    ):
        """
        Constructor of the Metaheuristic class
        """

        super().__init__(objfunc, search_strategy, params)

    def step(self, time_start=0, verbose=False):
        # Get the population of this generation
        population = self.search_strategy.population

        # Generate their parents
        parents = self.search_strategy.select_parents(
            population, progress=self.progress, history=self.best_history
        )

        # Evolve the selected parents
        offspring = self.search_strategy.perturb(
            parents, self.objfunc, progress=self.progress, history=self.best_history
        )

        # Select the individuals that remain for the next generation
        population = self.search_strategy.select_individuals(
            population, offspring, progress=self.progress, history=self.best_history
        )

        # Assign the newly generate population
        self.search_strategy.population = population

        # Get information about the algorithm to track it's progress
        best_individual, best_fitness = self.search_strategy.best_solution()
        self.search_strategy.update_params(progress=self.progress)

        # Store information
        self.best_history.append(best_individual)
        self.fit_history.append(best_fitness)

        return (best_individual, best_fitness)

    def step_info(self, start_time=0):
        print(f"Optimizing {self.objfunc.name} using {self.search_strategy.name}:")
        print(f"\tReal time Spent: {round(time.time() - start_time,2)} s")
        print(f"\tCPU time Spent:  {round(time.time() - start_time,2)} s")
        print(f"\tGeneration: {self.steps}")
        best_fitness = self.best_solution()[1]
        print(f"\tBest fitness: {best_fitness}")
        print(f"\tEvaluations of fitness: {self.objfunc.counter}")
        self.search_strategy.extra_step_info()
        print()

    def display_report(self, show_plots=True):
        print("Number of generations:", len(self.fit_history))
        print("Real time spent: ", round(self.real_time_spent, 5), "s", sep="")
        print("CPU time spent: ", round(self.cpu_time_spent, 5), "s", sep="")
        print("Number of fitness evaluations:", self.objfunc.counter)

        best_fitness = self.best_solution()[1]
        print("Best fitness:", best_fitness)

        if show_plots:
            # Plot fitness history
            plt.axhline(y=0, color="black", alpha=0.9)
            plt.axvline(x=0, color="black", alpha=0.9)
            plt.plot(self.fit_history, "blue")
            plt.xlabel("generations")
            plt.ylabel("fitness")
            plt.title(f"{self.search_strategy.name} fitness")
            plt.show()

        self.search_strategy.extra_report(show_plots)
