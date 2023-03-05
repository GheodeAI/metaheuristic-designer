import numpy as np
from matplotlib import pyplot as plt
import time
from .GeneralSearch import GeneralSearch

class MemeticSearch(GeneralSearch):
    """
    General framework for metaheuristic algorithms
    """
    
    def __init__(self, search_strategy, local_search, improve_choice, params):
        """
        Constructor of the Metaheuristic class
        """
        super().__init__(search_strategy, params)

        self.local_search = local_search
        self.improve_choice = improve_choice


    def _do_local_search(self, offspring):
        offspring_to_imp, off_idxs = self.improve_choice(offspring)

        to_improve = [offspring[i] for i in off_idxs]

        improved = self.local_search.perturb(to_improve)

        for idx, val in enumerate(off_idxs):
            offspring[val] = improved[idx]
        
        return offspring

    def step(self, time_start=0, verbose=False):
        """
        Performs a step in the algorithm
        """

        # Do a search step
        population = self.search_strategy.population
        
        parents, parent_idxs = self.search_strategy.select_parents(population, self.progress, self.best_history)

        offspring = self.search_strategy.perturb(parents, self.progress, self.best_history)        

        offspring = self._do_local_search(offspring)

        population = self.search_strategy.select_individuals(population, offspring, self.progress, self.best_history)


        self.search_strategy.population = population
        

        # Store best solution and update parameters
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
        self.update(self.steps, time)
        
        return (best_individual, best_fitness)