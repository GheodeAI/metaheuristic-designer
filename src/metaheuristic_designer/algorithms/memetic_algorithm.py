from __future__ import annotations
from copy import copy
import logging
from threading import local
from typing import Tuple, Any, Optional
from abc import ABC, abstractmethod
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from ..objective_function import ObjectiveFunc
from ..search_strategy import SearchStrategy
from ..population import Population
from ..parametrizable_mixin import ParametrizableMixin
from ..stopping_condition import StoppingCondition
from ..initializer import Initializer
from ..utils import NumpyEncoder
from ..parent_selection import ParentSelection
from ..algorithm import Algorithm

logger = logging.getLogger(__name__)

class MemeticAlgorithm(Algorithm):
    """

    Iterative search algorithm based on a standard loop with a local search improvement after perturbing
    the individuals. 

    Parameters
    ----------
    objfunc : ObjectiveFunc
        _description_
    search_strategy : SearchStrategy
        _description_
    local_search : SearchStrategy
        _description_
    improvement_selection : ParentSelection
        _description_
    local_search_frequency : int, optional
        _description_, by default 1
    local_search_depth : int, optional
        _description_, by default 1
    name : Optional[str], optional
        _description_, by default None
    init_info : bool, optional
        _description_, by default True
    verbose : bool, optional
        _description_, by default True
    v_timer : float, optional
        _description_, by default 1
    stop_cond : str, optional
        _description_, by default "time_limit"
    progress_metric : Optional[str], optional
        _description_, by default None
    ngen : int, optional
        _description_, by default 1000
    neval : int, optional
        _description_, by default 1e5
    time_limit : float, optional
        _description_, by default 60.0
    cpu_time_limit : float, optional
        _description_, by default 60.0
    fit_target : float, optional
        _description_, by default 1e-10
    patience : int, optional
        _description_, by default 100
    stopping_condition : Optional[StoppingCondition], optional
        _description_, by default None
    parallel : bool, optional
        _description_, by default False
    threads : int, optional
        _description_, by default 8
    """

    def __init__(
        self,
        objfunc: ObjectiveFunc,
        search_strategy: SearchStrategy,
        local_search: SearchStrategy,
        improvement_selection: ParentSelection,
        local_search_frequency: int = 1,
        local_search_depth: int = 1,
        lamarckian: bool = True,
        name: Optional[str] = None,
        init_info: bool = True,
        verbose: bool = True,
        v_timer: float = 1,
        stop_cond: str = "time_limit",
        progress_metric: Optional[str] = None,
        ngen: int = 1000,
        neval: int = 1e5,
        time_limit: float = 60.0,
        cpu_time_limit: float = 60.0,
        fit_target: float = 1e-10,
        patience: int = 100,
        stopping_condition: Optional[StoppingCondition] = None,
        parallel: bool = False,
        threads: int = 8,
    ):

        self.local_search = local_search
        self.improvement_selection = improvement_selection
        self.local_search_frequency = local_search_frequency
        self.local_search_depth = local_search_depth
        self.lamarckian = lamarckian

        if not local_search.operator.preserves_order:
            logger.warning("Local search implements an operator that doesn't preserve order (%s). The fitness calculation might be corrupted.", local_search.operator.name)

        if not local_search.survivor_sel.preserves_order:
            logger.warning("Local search implements a survivos selection method that doesn't preserve order (%s). The fitness calculation might be corrupted.", local_search.survivor_sel.name)

        self.local_search_counter = 0

        super().__init__(
            objfunc=objfunc,
            search_strategy=search_strategy,
            name=name,
            init_info=init_info,
            verbose=verbose,
            v_timer=v_timer,
            stop_cond=stop_cond,
            progress_metric=progress_metric,
            ngen=ngen,
            neval=neval,
            time_limit=time_limit,
            cpu_time_limit=cpu_time_limit,
            fit_target=fit_target,
            patience=patience,
            stopping_condition=stopping_condition,
            parallel=parallel,
            threads=threads,
        )
    
    @property
    def name(self):
        backup_name = f"Memetic {self.search_strategy.name}"
        return self._name if self._name else backup_name

    @name.setter
    def name(self, new_name: str):
        self._name = new_name

    def initialize(self):
        population = super().initialize()
        self.local_search.initialize(self.objfunc)
        return population

    def _do_local_search(self, offspring):
        # Select individuals to improve
        selected_to_improve = self.improvement_selection(offspring)
        chosen_idx = self.improvement_selection.last_selection_idx
        self._log_debug("Selected individuals to improve\n%s", selected_to_improve)

        next_selected_population = selected_to_improve
        for _ in range(self.local_search_depth):
            # Apply mutation to individuals
            mutated_offspring = self.local_search.perturb(next_selected_population)
            self._log_debug("Mutated inside local search\n%s", mutated_offspring)

            # Select the best individuals (ensure the population size is the same as the parents)
            improved_offspring = self.local_search.select_individuals(next_selected_population, mutated_offspring)
            self._log_debug("Selected inside local search\n%s", improved_offspring)

            # Assign improved individuals to the population
            offspring = offspring.apply_selection(improved_offspring, chosen_idx)
            offspring = self.search_strategy.evaluate_population(offspring, self.parallel, self.threads)
            next_selected_population = offspring

        self._log_debug("Applied local search\n%s", offspring)

        return offspring, chosen_idx

    def _log_debug(self, text, population):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(text, population.debug_repr())

    def step(self, population=None, time_start=0, verbose=False):
        self.local_search_counter += 1

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

        # Perform a local search on the best individuals
        offspring_memetic = copy(offspring)
        if self.local_search_counter % self.local_search_frequency == 0:
            offspring_memetic, chosen_idx = self._do_local_search(offspring_memetic)
            
            if not self.lamarckian:
                fitness_obtained = offspring_memetic.fitness
                offspring_memetic = offspring
                offspring_memetic.fitness[chosen_idx] = fitness_obtained[chosen_idx]


        self.local_search_counter += 1

        # Select the individuals that remain for the next generation
        new_population = self.search_strategy.select_individuals(population, offspring_memetic)
        self._log_debug("Selected\n%s", new_population)

        # Assign the newly generated population
        self.search_strategy.population = new_population

        # Get information about the algorithm to track it's progress
        self.search_strategy.step(self.progress)
        self._log_debug("Updated end\n%s", new_population)

        # Store information
        best_individual, best_fitness = self.search_strategy.best_solution()
        self.best_history.append(best_individual)
        self.fit_history.append(best_fitness)

        return new_population

    def get_state(self, show_fit_history: bool = False, show_gen_history: bool = False, show_population: bool = False):
        data = super().get_state(show_fit_history, show_gen_history, show_population)

        # Add parent selection method for local search
        data["improvement_selection"] = self.improvement_selection.get_state()

        # Add local search data
        local_search_data = self.local_search.get_state(show_population=False)
        local_search_data.pop("population", None)
        data["local_search_state"] = local_search_data

        # push search strategy data to the bottom
        search_strat_data = data.pop("search_strat_state", None)
        data["search_strat_state"] = search_strat_data

        return data

    def step_info(self, start_time=0):
        super().step_info(start_time)
        self.local_search.extra_step_info()
        print()
