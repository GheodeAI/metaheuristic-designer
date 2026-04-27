"""
Base class for the Algorithm module.

This module implements the main loop of the optimization algorithm using a search strategy.
"""

from __future__ import annotations
import logging
import sys
from typing import Tuple, Any, Optional
from abc import ABC, abstractmethod
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from .objective_function import ObjectiveFunc
from .search_strategy import SearchStrategy
from .population import Population
from .parametrizable_mixin import ParametrizableMixin
from .stopping_condition import StoppingCondition
from .initializer import Initializer
from .utils import NumpyEncoder

logger = logging.getLogger(__name__)


class Algorithm(ABC):
    """
    Abstract Algorithm class.

    This class defines the structure of all optimization algorithms.

    Parameters
    ----------

    objfunc: ObjectiveFunc
        Objective function to be optimized.
    search_strategy: SearchStrategy
        Search strategy that will iteratively optimize the function.
    name: str, optional
        Name that will be displayed when showing the algorithm.
    """

    def __init__(
        self,
        objfunc: ObjectiveFunc,
        search_strategy: SearchStrategy,
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
        super().__init__()

        self.search_strategy = search_strategy
        self.objfunc = objfunc
        self._name = name

        # Verbose parameters
        self.show_init_info = init_info
        self.verbose = verbose
        self.v_timer = v_timer

        if stopping_condition is None:
            stopping_condition = StoppingCondition(
                condition_str=stop_cond,
                progress_metric_str=progress_metric,
                time_limit=time_limit,
                cpu_time_limit=cpu_time_limit,
                target_fitness=fit_target,
                max_evaluations=neval,
                max_iterations=ngen,
                max_patience=patience,
                optimization_mode=objfunc.mode,
            )
        self.stopping_condition = stopping_condition

        # Parallel parameters
        self.parallel = parallel
        self.threads = threads

        # Metrics
        self.fit_history = []
        self.best_history = []
        self.progress = 0
        self.ended = False
        self.prev_best_fitness = None
        self.cpu_time_spent = 0
        self.real_time_spent = 0
        self.converged_steps = 0

    @property
    def name(self) -> str:
        return self._name if self._name else self.search_strategy.name

    @name.setter
    def name(self, new_name: str):
        self._name = new_name

    @property
    def initializer(self) -> Initializer:
        return self.search_strategy.initializer

    @initializer.setter
    def initializer(self, new_initializer):
        self.search_strategy.initializer = new_initializer

    @property
    def iterations(self) -> int:
        return self.stopping_condition.iterations

    @property
    def evaluations(self) -> int:
        return self.objfunc.counter

    @property
    def patience_left(self) -> int:
        return self.stopping_condition.patience_left

    @property
    def population(self) -> Population:
        return self.search_strategy.population

    def best_solution(self, decoded: bool = False) -> Tuple[Any, float]:
        """
        Returns the best solution so far in the population.

        Returns
        -------
        best_solution: Tuple[Any, float]
            A pair of the best individual with its fitness.
        """

        return self.search_strategy.best_solution(decoded)

    def restart(self, restart_objfunc: bool = True):
        """
        Resets the internal values of the algorithm and the number of evaluations of the fitness function.
        """

        self.fit_history = []
        self.best_history = []
        if restart_objfunc:
            self.objfunc.restart()
        self.stopping_condition.restart()

        logger.debug("Reset the data of the algorithm.")

    def save_solution(self, file_name: str = "solution.csv"):
        """
        Save the result of an execution to a csv file in disk.

        Parameters
        ----------

        file_name: str
            Path to the file where the solution will be stored.
        """

        ind, _ = self.search_strategy.best_solution(decoded=False)
        np.savetxt(file_name, ind.reshape([1, -1]), delimiter=",")
        logger.info("Successfully saved the optimization history to %s", file_name)

    def update(self, skip_step: bool = False):
        """
        Updates the attributes of the optimization algorithm.
        This function should be called once per iteration of the algorithm.

        Parameters
        ----------
        real_time_start: float
            The time in seconds that passed since the algorithm was executed.
        cpu_time_start: float
            The time in seconds that the CPU has executed code in this algorithm.
        pass_step: bool
            Whether to increment the iteration counter or not.
        """

        self.stopping_condition.step(self.population, skip_step)

        self.progress = self.stopping_condition.get_progress()

    def initialize(self, reset_objfunc=True) -> Population:
        """
        Initializes the optimization algorithm.

        Returns
        -------
        initial_population: Population
            The first set of individuals generated in order to perform the optimization.
        """

        self.restart(reset_objfunc)
        initial_population = self.search_strategy.initialize(self.objfunc)
        initial_population = self.search_strategy.evaluate_population(initial_population, self.parallel, self.threads)
        self.search_strategy.population = initial_population

        return initial_population

    @abstractmethod
    def step(self, population=None) -> Population:
        """
        Performs an iteration of the algorithm.

        Parameters
        ----------
        population: Population, optional
            Population to evolve in the next generation. By default use the result from
            the previous step contained in the search strategy class.
        time_start: float, optional
            Indicates to the algorithm how much time has already passed.
        verbose: bool, optional
            Indicates whether to show the status of the algorithm or not.

        Returns
        -------
        current_population: Population
            The new population obtained in this iteration of the algorithm.
        """

    def optimize(self, initialize=True) -> Population:
        """
        Execute the algorithm to get the best solution possible along with its evaluation.
        It will initialize the algorithm and repeat steps of the algorithm until the
        stopping condition is met.

        Returns
        -------
        current_population: Population
            Population of the best individuals found by the algorithm.
        """

        if self.verbose and self.show_init_info:
            self.init_info()

        self.steps = 0

        # initialize clocks
        real_time_start = time.time()
        cpu_time_start = time.process_time()
        self.stopping_condition.real_time_start = real_time_start
        self.stopping_condition.cpu_time_start = cpu_time_start
        display_timer = time.time()

        # Initialize search strategy
        logger.info("Generating initial solutions...")
        if initialize:
            population = self.initialize()
        else:
            population = self.search_strategy.population

        # Search until the stopping condition is met
        self.update(skip_step=True)
        self.step_info(real_time_start)

        logger.info("Starting main optimization loop...")
        while not self.stopping_condition.is_finished(self.search_strategy.finish):
            logger.info("Started iteration %d...", self.iterations)

            population = self.step(population=population)
            self.update()

            # Display information
            if time.time() - display_timer > self.v_timer:
                self.step_info(real_time_start)
                display_timer = time.time()

        # Store the time spent optimizing
        self.real_time_spent = time.time() - real_time_start
        self.cpu_time_spent = time.process_time() - cpu_time_start

        logger.info("Optimization finished.")

        return population

    def get_state(self, show_fit_history: bool = False, show_gen_history: bool = False, show_population: bool = False) -> dict:
        """
        Gets the current state of the algorithm as a dictionary.

        Parameters
        ----------
        show_best_solution: bool, optional
            Save the best solution found by the algorithm.
        show_fit_history: bool, optional
            Save the fitness of the best individual of each iteration.
        show_gen_history: bool, optional
            Save the best individual for each iteration.
        show_pop: bool, optional
            Save the entire population of the last iteration.
        show_pop_details:bool, optional
            Save the detailed information of each individual.

        Returns
        -------
        state: dict
            The complete state of the algorithm.
        """

        data = {
            "name": self.name,
            "objfunc": self.objfunc.name,
            "ended": self.ended,
            "progress": self.progress,
            "generation": self.steps,
            "evaluations": self.objfunc.counter,
            "real_time_spent": self.real_time_spent,
            "cpu_time_spent": self.cpu_time_spent,
            "params": self.params,
        }

        if show_fit_history:
            data["fit_history"] = self.fit_history

        if show_gen_history:
            data["best_history"] = self.best_history

        data["search_strategy"] = self.search_strategy.get_state(show_population)

        return data

    def store_state(
        self,
        file_name: str = "dumped_state.json",
        readable: bool = False,
        show_fit_history: bool = False,
        show_gen_history: bool = False,
        show_population: bool = False,
    ):
        """
        Dumps the current state of the algorithm to a JSON file.

        Parameters
        ----------
        file_name: str
            Path to the file where the json file will be stored.
        readable: bool, optional
            Indent the JSON file to make it human-readable (comes at the cost of a higher file size).
        show_best_solution: bool, optional
            Save the best solution found by the algorithm.
        show_fit_history: bool, optional
            Save the fitness of the best individual of each iteration.
        show_gen_history: bool, optional
            Save the best individual for each iteration.
        show_pop: bool, optional
            Save the entire population of the last iteration.
        show_pop_details:bool, optional
            Save the detailed information of each individual.
        """

        dumped = json.dumps(self.get_state(show_fit_history, show_gen_history, show_population), cls=NumpyEncoder, indent=4 if readable else None)

        with open(file_name, "w", encoding="utf-8") as fp:
            fp.write(dumped)

    def init_info(self):
        print(f"Initializing optimization of {self.objfunc.name} using {self.search_strategy.name}")
        print(f"-----------------------------{'-'*len(self.objfunc.name)}-------{'-'*len(self.search_strategy.name)}")
        print()

    def step_info(self, start_time: float = 0):
        """
        Displays information about the current state of the algorithm.

        Parameters
        ----------
        time_start: float, optional
            Indicates to the algorithm how much time has already passed.
        """

        logger.debug("Finished iteration %d.", self.iterations)

        if self.verbose:
            print(f"Optimizing {self.objfunc.name} using {self.name}:")
            print(f"\tReal time Spent: {time.time() - start_time:0.3f} s")
            print(f"\tCPU time Spent:  {time.time() - start_time:0.3f} s")
            print(f"\tGeneration: {self.iterations}")
            _, best_fitness = self.best_solution()
            print(f"\tBest fitness: {best_fitness}")
            print(f"\tEvaluations of fitness: {self.objfunc.counter}")
            print()
            self.search_strategy.extra_step_info()
            print()

    def display_report(self, show_plots: bool = True):
        """
        Shows a summary of the execution of the algorithm.

        Parameters
        ----------
        show_plots: bool, optional
            Whether to display plots about the algorithm or not.
        """

        print("Number of generations:", len(self.fit_history))
        print("Real time spent: ", round(self.real_time_spent, 5), "s", sep="")
        print("CPU time spent: ", round(self.cpu_time_spent, 5), "s", sep="")
        print("Number of fitness evaluations:", self.objfunc.counter)

        best_fitness = self.best_solution()[1]
        print("Best fitness:", best_fitness)

        if show_plots:
            # Plot fitness history
            _, ax = plt.subplots()
            ax.plot(self.fit_history, color="blue", zorder=3)
            _xlim = ax.get_xlim()
            _ylim = ax.get_ylim()
            ax.axhline(y=0, color="black", alpha=0.9)
            ax.axvline(x=0, color="black", alpha=0.9)
            ax.set_xlim(_xlim)
            ax.set_ylim(_ylim)
            ax.set(xlabel="Generations", ylabel="Fitness", title=f"{self.search_strategy.name} fitness")
            ax.grid()
            logger.debug("Generated summary plot.")
            plt.show()

        self.search_strategy.extra_report(show_plots)
