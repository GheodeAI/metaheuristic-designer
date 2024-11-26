from __future__ import annotations
from ..Algorithm import Algorithm


class MemeticAlgorithm(Algorithm):
    """
    Framework for algorithms that incorporate a combination of a main search strategy and a local search step.

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

    def __init__(
        self,
        objfunc,
        search_strategy,
        local_search,
        improve_choice,
        params=None,
        name=None,
    ):
        """
        Constructor of the Metaheuristic class
        """

        self.local_search = local_search
        self.improve_choice = improve_choice

        super().__init__(objfunc, search_strategy, params, name)

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
        # Select individuals to improve
        selected_to_improve = self.improve_choice(offspring)
        chosen_idx = self.improve_choice.last_selection_idx

        # Apply mutation to individuals
        mutated_offspring = self.local_search.perturb(selected_to_improve)

        # Select the best individuals (ensure the population size is the same as the parents)
        improved_offspring = self.local_search.select_individuals(selected_to_improve, mutated_offspring)

        # Assign improved individuals to the population
        offspring = offspring.apply_selection(improved_offspring, chosen_idx)
        offspring = self.search_strategy.evaluate_population(offspring, self.parallel, self.threads)

        return offspring

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

        # Perform a local search on the best individuals
        offspring = self._do_local_search(offspring)

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
