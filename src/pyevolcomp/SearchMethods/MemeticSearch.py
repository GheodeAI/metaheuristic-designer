from __future__ import annotations
import numpy as np
from matplotlib import pyplot as plt
import time
from ..Search import Search


class MemeticSearch(Search):
    """
    General framework for metaheuristic algorithms
    """

    def __init__(self, objfunc, search_strategy, local_search, improve_choice, pop_init = None, params = None):
        """
        Constructor of the Metaheuristic class
        """
        super().__init__(objfunc, search_strategy, pop_init, params)

        self.local_search = local_search
        self.improve_choice = improve_choice

    def initialize(self):
        """
        Generates a random population of individuals
        """

        super().initialize()
        initial_population = self.pop_init.generate_population(self.objfunc)[:1]
        self.local_search.initialize(initial_population)

    def _do_local_search(self, offspring):
        offspring_to_imp, off_idxs = self.improve_choice(offspring)

        to_improve = [offspring[i] for i in off_idxs]

        improved = self.local_search.perturb(to_improve, self.objfunc)

        for idx, val in enumerate(off_idxs):
            offspring[val] = improved[idx]

        current_best = max(improved, key=lambda x: x.fitness)
        if self.search_strategy.best.fitness < current_best.fitness:
            self.search_strategy.best = current_best

        return offspring

    def step(self, time_start=0, verbose=False):
        """
        Performs a step in the algorithm
        """

        # Do a search step
        population = self.search_strategy.population

        parents, parent_idxs = self.search_strategy.select_parents(population, self.progress, self.best_history)

        offspring = self.search_strategy.perturb(parents, self.objfunc, self.progress, self.best_history)

        offspring = self._do_local_search(offspring)

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
        self.update(self.steps, time_start)

        return (best_individual, best_fitness)
    
    def get_state(self, objfunc):
        """
        Gets the current state of the algorithm as a dictionary
        """
        
        data = super().get_state(objfunc)

        # Add parent selection method for local search
        data["improve_selection"] = self.improve_choice.get_state()

        # Add local search data
        local_search_data = self.local_search.get_state()
        local_search_data.pop("population", None)
        local_search_data.pop("best_individual", None)
        data["local_search_state"] = local_search_data

        # push search strategy data to the bottom
        search_strat_data = data.pop("search_strat_state", None)
        data["search_strat_state"] = search_strat_data
        
        return data

    def step_info(self, objfunc, start_time):
        """
        Displays information about the current state of the algotithm
        """

        print(f"Optimizing {self.objfunc.name} using {self.search_strategy.name}+{self.local_search.name}:")
        print(f"\tReal time Spent: {round(time.time() - start_time,2)} s")
        print(f"\tCPU time Spent:  {round(time.time() - start_time,2)} s")
        print(f"\tGeneration: {self.steps}")
        best_fitness = self.best_solution()[1]
        print(f"\tBest fitness: {best_fitness}")
        print(f"\tEvaluations of fitness: {self.objfunc.counter}")
        self.search_strategy.extra_step_info()
        self.local_search.extra_step_info()
        print()

    def display_report(self, show_plots=True):
        """
        Shows a summary of the execution of the algorithm
        """

        # Print Info
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
