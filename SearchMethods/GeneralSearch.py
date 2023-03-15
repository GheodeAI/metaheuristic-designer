import numpy as np
from matplotlib import pyplot as plt
from typing import Union
from ..ParamScheduler import ParamScheduler
from ..BaseAlgorithm import BaseAlgorithm
from ..BaseSearch import BaseSearch
import time


class GeneralSearch(BaseSearch):
    """
    General framework for metaheuristic algorithms
    """
    
    def __init__(self, search_strategy: BaseAlgorithm, params: Union[ParamScheduler, dict]):
        """
        Constructor of the Metaheuristic class
        """

        super().__init__(search_strategy, params)
        

    def step(self, objfunc, time_start=0, verbose=False):
        """
        Performs a step in the algorithm
        """

        # Do a search step
        population = self.search_strategy.population
        
        parents, _ = self.search_strategy.select_parents(population, self.progress, self.best_history)

        offspring = self.search_strategy.perturb(parents, objfunc, self.progress, self.best_history)

        population = self.search_strategy.select_individuals(population, offspring, self.progress, self.best_history)

        self.search_strategy.population = population
        
        best_individual, best_fitness = self.search_strategy.best_solution()
        self.search_strategy.update_params(self.progress)
        self.steps += 1
            
        # Store information
        self.best_history.append(best_individual)
        self.fit_history.append(best_fitness)

        # Display information
        if verbose:
            self.step_info(time_start)
        
        # Update internal state
        self.update(self.steps, time_start, objfunc)
        
        return (best_individual, best_fitness)
        
    
    def step_info(self, objfunc, start_time):
        """
        Displays information about the current state of the algotithm
        """

        print(f"Optimizing {objfunc.name} using {self.search_strategy.name}:")
        print(f"\tTime Spent {round(time.time() - start_time,2)} s")
        print(f"\tGeneration: {self.steps}")
        best_fitness = self.best_solution()[1]
        print(f"\tBest fitness: {best_fitness}")
        print(f"\tEvaluations of fitness: {objfunc.counter}")
        self.search_strategy.extra_step_info()
        print()
    
    
    def display_report(self, objfunc, show_plots=True):
        """
        Shows a summary of the execution of the algorithm
        """
        
        # Print Info
        print("Number of generations:", len(self.fit_history))
        print("Real time spent: ", round(self.real_time_spent, 5), "s", sep="")
        print("CPU time spent: ", round(self.time_spent, 5), "s", sep="")
        print("Number of fitness evaluations:", objfunc.counter)
        
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