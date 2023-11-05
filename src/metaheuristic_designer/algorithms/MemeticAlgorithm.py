from __future__ import annotations
import time
from matplotlib import pyplot as plt
from ..Algorithm import Algorithm


class MemeticAlgorithm(Algorithm):
    """
    General framework for metaheuristic algorithms

    Parameters
    ----------

    objfunc: ObjectiveFunc
        Objective function to be optimized.
    search_strategy: Algorithm
        Search strategy that will iteratively optimize the function.
    local_search: Algorithm
        Search strategy that will improve a selection of the individuals.
    improve_choice: SelectionMethod
        Method used to select the individuals that will be improved
    params: ParamScheduler or dict, optional
        Dictionary of parameters to define the stopping condition and output of the algorithm.
    """

    def __init__(self, objfunc, search_strategy, local_search, improve_choice, params=None, name=None):
        """
        Constructor of the Metaheuristic class
        """

        super().__init__(objfunc, search_strategy, params, name)

        self.local_search = local_search
        self.improve_choice = improve_choice

    @property
    def name(self):
        backup_name = f"Memetic {self.search_strategy.name}"
        return self._name if self._name else backup_name
    
    @name.setter
    def name(self, new_name: str):
        self._name = new_name

    def initialize(self):
        super().initialize()
        self.local_search.initialize(self.objfunc)

    def _do_local_search(self, offspring):
        offspring_ids = [indiv.id for indiv in offspring]

        offspring_to_imp = self.improve_choice(offspring)
        off_idxs = [offspring_ids.index(indiv.id) for indiv in offspring_to_imp]

        to_improve = [offspring[i] for i in off_idxs]

        improved = self.local_search.perturb(to_improve, self.objfunc)

        for idx, val in enumerate(off_idxs):
            offspring[val] = improved[idx]

        current_best = max(improved, key=lambda x: x.fitness)
        if self.search_strategy.best.fitness < current_best.fitness:
            self.search_strategy.best = current_best

        return offspring

    def step(self, time_start=0, verbose=False):
        population = self.search_strategy.population

        parents = self.search_strategy.select_parents(population, progress=self.progress, history=self.best_history)

        offspring = self.search_strategy.perturb(parents, self.objfunc, progress=self.progress, history=self.best_history)

        offspring = self._do_local_search(offspring)

        population = self.search_strategy.select_individuals(population, offspring, progress=self.progress, history=self.best_history)

        self.search_strategy.population = population

        best_individual, best_fitness = self.search_strategy.best_solution()
        self.search_strategy.update_params(progress=self.progress)

        # Store information
        self.best_history.append(best_individual)
        self.fit_history.append(best_fitness)

        # Display information
        if verbose:
            self.step_info(time_start)

        # Update internal state
        self.update(self.steps, time_start)

        return (best_individual, best_fitness)

    def get_state(
        self,
        show_best_solution: bool = True,
        show_fit_history: bool = False,
        show_gen_history: bool = False,
        show_pop: bool = False,
        show_pop_details: bool = False,
    ):
        data = super().get_state(
            show_best_solution,
            show_fit_history,
            show_gen_history,
            show_pop,
            show_pop_details,
        )

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

    def step_info(self, start_time=0):
        super().step_info(start_time)
        self.local_search.extra_step_info()
        print()
