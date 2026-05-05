"""
Base class for the Algorithm module.

This module implements the main loop of the optimization algorithm using a search strategy.
"""

from __future__ import annotations
import logging
from typing import Tuple, Any, Optional
import json
import numpy as np
import signal

from .history_tracker import HistoryTracker
from .reporters import create_reporter
from .reporter import Reporter
from .reporters import VerboseReporter
from .objective_function import ObjectiveFunc
from .search_strategy import SearchStrategy
from .population import Population
from .stopping_condition import StoppingCondition
from .initializer import Initializer
from .checkpointer import Checkpointer
from .utils import NumpyEncoder, VectorLike

logger = logging.getLogger(__name__)


class TerminationException(Exception):
    """
    Custom exception to handle SIGTERM
    """


class Algorithm:
    """
    Abstract Algorithm class.

    This class defines the structure of all iterative optimization algorithms.

    Parameters
    ----------
    objfunc : ObjectiveFunc
        _description_
    search_strategy : SearchStrategy
        _description_
    name : Optional[str], optional
        _description_, by default None
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
    verbose_timer : float, optional
        _description_, by default 0.5
    stopping_condition : Optional[StoppingCondition], optional
        _description_, by default None
    reporter : Optional[Reporter], optional
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
        name: Optional[str] = None,
        stop_cond: str = "real_time_limit",
        progress_metric: Optional[str] = None,
        max_iterations: int = 1000,
        max_evaluations: int = 1e5,
        real_time_limit: float = 60.0,
        cpu_time_limit: float = 60.0,
        objective_target: float = 1e-10,
        max_patience: int = 100,
        verbose_timer: float = 0.5,
        track_median: bool = False,
        track_worst: bool = False,
        track_full_objective: bool = False,
        track_full_population: bool = False,
        track_diversity: bool = False,
        checkpoint_file: Optional[str] = None,
        checkpoint_time_frequency: Optional[float] = None,
        checkpoint_iteration_frequency: Optional[float] = None,
        stopping_condition: Optional[StoppingCondition] = None,
        reporter: Optional[str | Reporter] = None,
        history_tracker: Optional[HistoryTracker] = None,
        checkpointer: Optional[Checkpointer] = None,
        parallel: bool = False,
        threads: int = 8,
    ):
        super().__init__()

        self.search_strategy = search_strategy
        self.objfunc = objfunc
        if name is None:
            name = self.search_strategy.name
        self.name = name

        # Parallel parameters
        self.parallel = parallel
        self.threads = threads

        # Stopping conditions
        if stopping_condition is None:
            stopping_condition = StoppingCondition(
                condition_str=stop_cond,
                progress_metric_str=progress_metric,
                real_time_limit=real_time_limit,
                cpu_time_limit=cpu_time_limit,
                objective_target=objective_target,
                max_evaluations=max_evaluations,
                max_iterations=max_iterations,
                max_patience=max_patience,
                optimization_mode=objfunc.mode,
            )
        self.stopping_condition = stopping_condition

        # Reporter
        if reporter is None:
            reporter = VerboseReporter(verbose_timer=verbose_timer)
        elif isinstance(reporter, str):
            reporter = create_reporter(reporter)
        self.reporter = reporter

        # History Tracker
        if history_tracker is None:
            history_tracker = HistoryTracker(
                track_best=True,
                track_median=track_median,
                track_worst=track_worst,
                track_full_objective=track_full_objective,
                track_full_population=track_full_population,
                track_diversity=track_diversity,
            )
        self.history_tracker = history_tracker

        # Checkpointer
        if checkpointer is not None or checkpoint_file is not None:
            if checkpointer is None:
                checkpointer = Checkpointer(
                    checkpoint_file=checkpoint_file,
                    iteration_frequency=checkpoint_iteration_frequency,
                    time_frequency=checkpoint_time_frequency,
                )
            self.checkpointer = checkpointer
        else:
            logger.info("Checkpointing is disabled since no checkpoint file was indicated.")
            self.checkpointer = None

        self._stop_requested = False
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        self._stop_requested = True

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
    def progress(self) -> float:
        return self.stopping_condition.get_progress()

    @property
    def population(self) -> Population:
        return self.search_strategy.population

    def gather_parameters(self):
        return self.search_strategy.gather_parameters()

    def best_solution(self) -> Tuple[Any, float]:
        """
        Returns the best solution so far in the population.

        Returns
        -------
        best_solution: Tuple[Any, float]
            A pair of the best individual with its objective value.
        """

        return self.search_strategy.best_solution()

    def best_individual(self) -> Tuple[VectorLike, float]:
        """
        Returns the best individual so far in the population.

        Returns
        -------
        best_solution: Tuple[VectorLike, float]
            A pair of the best individual with its fitness.
        """

        return self.search_strategy.best_individual()

    def restart(self, restart_objfunc: bool = True):
        """
        Resets the internal values of the algorithm and the number of evaluations of the fitness function.
        """

        if restart_objfunc:
            self.objfunc.restart()
        self.stopping_condition.restart()
        self.history_tracker.restart()
        if self.checkpointer is not None:
            self.checkpointer.restart()

        logger.debug("Reset the data of the algorithm.")

    def save_solution(self, file_name: str = "solution.csv"):
        """
        Save the result of an execution to a csv file in disk.

        Parameters
        ----------

        file_name: str
            Path to the file where the solution will be stored.
        """

        individual, _ = self.search_strategy.best_solution(problem_space=True)
        np.savetxt(file_name, individual.reshape([1, -1]), delimiter=",")
        logger.info("Successfully saved the optimization history to %s", file_name)

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

    def _log_debug(self, text, population):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(text, population.debug_repr())

    def step(self, population=None):
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

        # Select the individuals that remain for the next generation
        new_population = self.search_strategy.select_individuals(population, offspring)
        self._log_debug("Selected\n%s", new_population)

        self.search_strategy.population = new_population

        # Update in cascade all the objects involved in the optimization
        self.search_strategy.step(progress=self.progress)
        self._log_debug("Updated end\n%s", new_population)

        return new_population

    def optimize(self, resume=False) -> Population:
        """
        Execute the algorithm to get the best solution possible along with its evaluation.
        It will initialize the algorithm and repeat steps of the algorithm until the
        stopping condition is met.

        Returns
        -------
        current_population: Population
            Population of the best individuals found by the algorithm.
        """

        self.reporter.log_init(self)

        # initialize clocks
        if not resume:
            self.stopping_condition.restart()

        # Initialize search strategy and record initial values.
        logger.info("Generating initial solutions...")
        population = self.population if resume else self.initialize()
        self.history_tracker.step(self)

        # Search until the stopping condition is met
        logger.info("Starting main optimization loop...")
        try:
            while not self.stopping_condition.is_finished(self.search_strategy.finish):
                logger.info("Started iteration %d...", self.iterations)

                population = self.step(population=population)

                self.stopping_condition.step(self.population)
                self.reporter.log_step(self)
                self.history_tracker.step(self)

                if self.checkpointer is not None:
                    self.checkpointer.checkpoint(self)

                if self._stop_requested:
                    raise TerminationException

        except (KeyboardInterrupt, TerminationException) as e:
            if self.checkpointer is not None:
                self.checkpointer.save(self)
            self.reporter.log_end(self)
            logger.info("Optimization aborted by an OS signal.")
            raise e

        self.reporter.log_end(self)
        logger.info("Optimization finished.")

        return population

    def get_state(self, store_population: bool = False) -> dict:
        """
        Gets the current state of the algorithm as a dictionary.

        Parameters
        ----------
        store_population: bool, optional
            Save the entire population of the last iteration.

        Returns
        -------
        state: dict
            The complete state of the algorithm.
        """

        data = {
            "class_name": self.__class__.__name__,
            "name": self.name,
            "objfunc": self.objfunc.get_state(),
            "stopping_condition": self.stopping_condition.get_state(),
            "search_strategy": self.search_strategy.get_state(store_population),
            "history": self.history_tracker.get_state(),
        }

        return data

    def store_state(
        self,
        file_name: str = "dumped_state.json",
        readable: bool = False,
    ):
        """
        Dumps the current state of the algorithm to a JSON file.

        Parameters
        ----------
        file_name: str
            Path to the file where the json file will be stored.
        readable: bool, optional
            Indent the JSON file to make it human-readable (comes at the cost of a higher file size).
        """

        dumped = json.dumps(self.get_state(), cls=NumpyEncoder, indent=4 if readable else None)

        with open(file_name, "w", encoding="utf-8") as fp:
            fp.write(dumped)
