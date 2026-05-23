"""
Base class for the Algorithm module.

This module implements the main loop of the optimization algorithm using a search strategy.
"""

from __future__ import annotations
import logging
from typing import Tuple, Any, Optional
import json
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
    Orchestrates a complete optimization run.

    An :class:`Algorithm` combines a :class:`ObjectiveFunc` with a
    :class:`SearchStrategy` and manages the iteration loop, stopping
    conditions, reporting, history tracking, and checkpointing.

    All runtime settings can be supplied as plain keyword arguments
    (e.g., ``max_iterations=200``) or as pre-built objects
    (:class:`StoppingCondition`, :class:`Reporter`, etc.).  The
    keyword-argument style is convenient for quick experiments; the
    object-based style gives finer control and reusability.

    Parameters
    ----------
    objfunc : ObjectiveFunc
        The objective function to optimize.
    search_strategy : SearchStrategy
        Strategy that defines one iteration of the algorithm.
    name : str, optional
        Display name for the algorithm (defaults to the strategy's name).
    stop_cond : str, optional
        Expression that defines the stopping condition (see
        :class:`StoppingCondition`). Default ``"real_time_limit"``.
    progress_metric : str, optional
        Token used to compute the 0-1 progress value for parameter
        schedules. Defaults to the same tokens as *stop_cond*.
    max_iterations : int, optional
        Maximum number of iterations (default 1000).
    max_evaluations : int, optional
        Maximum number of objective evaluations (default 1e5).
    real_time_limit : float, optional
        Wall-clock time limit in seconds (default 60).
    cpu_time_limit : float, optional
        CPU time limit in seconds (default 60).
    objective_target : float, optional
        Target value for the raw objective (default 1e-10).
    max_patience : int, optional
        Iterations without improvement before ``convergence`` stops
        (default 100).
    verbose_timer : float, optional
        Interval in seconds between prints when using the default
        :class:`VerboseReporter` (default 0.5).
    track_median / track_worst / track_full_objective / track_full_population / track_diversity : bool, optional
        Flags forwarded to the :class:`HistoryTracker` when one is not
        supplied explicitly.
    checkpoint_file / checkpoint_time_frequency / checkpoint_iteration_frequency : optional
        Arguments used to construct a :class:`Checkpointer` when
        *checkpointer* is not given.
    stopping_condition : StoppingCondition, optional
        Explicit stopping condition object.
    reporter : str or Reporter, optional
        Reporter instance or name (``"tqdm"``, ``"silent"``, ``"verbose"``).
    history_tracker : HistoryTracker, optional
        Explicit history tracker.
    checkpointer : Checkpointer, optional
        Explicit checkpointer.
    parallel : bool, optional
        Whether to evaluate the population in parallel (currently
        reserved for future use).
    threads : int, optional
        Number of threads for parallel evaluation (reserved).
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
        self.population = None
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handles SIGINT Os-level signals.

        Parameters
        ----------
        signum : Signal number identifier (unused)
        frame : Frame of the signal (unused)
        """

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

    def gather_parameters(self) -> dict:
        """
        Collect the current parameters of the underlying search strategy.

        Returns
        -------
        dict
            A dictionary of parameter names and their current values.
        """

        param_dict = {}
        if self.population is not None:
            param_dict.update({f"{self.population.encoding.name}.{k}": v for k, v in self.population.encoding.gather_params().items()})
        param_dict.update(self.search_strategy.gather_parameters())
        return param_dict

    def best_solution(self) -> Tuple[Any, float]:
        """
        Return the best decoded solution and its raw objective value.

        Returns
        -------
        best_solution: Tuple[Any, float]
            A pair of the best individual with its objective value.
        """

        return self.population.best_solution()

    def best_individual(self) -> Tuple[VectorLike, float]:
        """
        Return the best genotype and its internal fitness value.

        Returns
        -------
        best_solution: Tuple[VectorLike, float]
            A pair of the best individual with its fitness.
        """

        return self.population.best_individual()

    def restart(self, restart_objfunc: bool = True):
        """
        Reset internal counters and, optionally, the objective function.

        Parameters
        ----------
        restart_objfunc : bool, optional
            If ``True``, also reset the objective function's evaluation
            counter.
        """

        if restart_objfunc:
            self.objfunc.restart()
        self.stopping_condition.restart()
        self.history_tracker.restart()
        if self.checkpointer is not None:
            self.checkpointer.restart()

        logger.debug("Reset the data of the algorithm.")

    def _log_debug(self, text, population):
        """
        Util for debugging population info.
        """

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(text, population.debug_repr())

    def resume(self) -> Population:
        """
        Resume an interrupted run from the last checkpoint.

        Returns
        -------
        Population
            The final population after the run completes.
        """

        return self.optimize(resume=True)

    def initialize(self) -> Population:
        """Generates the initial population from the search strategy.

        This method stores the population in the `.population` attribute and returns it.

        Returns
        -------
        initial_population
            The initial population generated.
        """

        self.population = self.search_strategy.initialize(self.objfunc)
        return self.population

    def step(self, prev_population: Population):
        """Performs a single step of the optimization algorithm.

        This method stores the population in the `.population` attribute and returns it.

        Parameters
        ----------
        prev_population : Population
            Population to be improved in this step of the optimization.

        Returns
        -------
        population
            The improved next population.
        """

        self.population = self.search_strategy.step(prev_population=prev_population)
        self.update()
        return self.population

    def update(self):
        """Updates the internal state of the algorithm."""
        self.stopping_condition.update(self.population)
        self.reporter.log_step(self)
        self.history_tracker.update(self)

        self.search_strategy.update(self.stopping_condition.get_progress())

        if self.checkpointer is not None:
            self.checkpointer.checkpoint(self)

    def optimize(self, resume: bool = False) -> Population:
        """
        Run the optimization loop until a stopping condition is met.

        Parameters
        ----------
        resume : bool, optional
            If ``True``, do not reset the algorithm state - continue
            from the current population and counters.

        Returns
        -------
        Population
            The final population.

        Raises
        ------
        KeyboardInterrupt, TerminationException
            If the process is interrupted, a checkpoint is attempted
            before re-raising.
        """

        self.reporter.log_init(self)

        # Initialize search strategy and record initial values.
        logger.info("Generating initial solutions...")

        # initialize clocks
        if not resume:
            self.restart()
            self.stopping_condition.restart()
            self.population = self.initialize()

        self.history_tracker.update(self)

        # Search until the stopping condition is met
        logger.info("Starting main optimization loop...")
        try:
            while not self.stopping_condition.is_finished(self.search_strategy.finish):
                logger.debug("Started iteration %d...", self.iterations)

                self.population = self.step(self.population)

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

        return self.population

    def get_state(self, store_population: bool = False) -> dict:
        """
        Serializes the current algorithm state to a dictionary.

        Parameters
        ----------
        store_population : bool, optional
            If ``True``, include the complete genotype matrix.

        Returns
        -------
        dict
            Dictionary representation of the algorithm state.
        """

        data = {
            "class_name": self.__class__.__name__,
            "name": self.name,
            "objfunc": self.objfunc.get_state(),
            "stopping_condition": self.stopping_condition.get_state(),
            "search_strategy": self.search_strategy.get_state(),
            "population": self.population.get_state() if store_population else None,
            "history": self.history_tracker.get_state(),
        }

        return data

    def store_state(
        self,
        file_name: str = "dumped_state.json",
        readable: bool = False,
    ):
        """
        Serialize the current algorithm state to a JSON file.

        Parameters
        ----------
        file_name : str, optional
            Destination path (default ``"dumped_state.json"``).
        readable : bool, optional
            If ``True``, produce indented JSON (larger but human
            readable).
        """

        dumped = json.dumps(self.get_state(), cls=NumpyEncoder, indent=4 if readable else None)

        with open(file_name, "w", encoding="utf-8") as fp:
            fp.write(dumped)

    def to_pandas(self):
        """
        Shorthand for ``self.history_tracker.to_pandas()``.

        Returns
        -------
        pandas.DataFrame
            Per-iteration summary of tracked metrics.
        """
        return self.history_tracker.to_pandas()

    def to_pandas_full_objective(self):
        """
        Shorthand for ``self.history_tracker.to_pandas_full_objective()``.

        Returns
        -------
        pandas.DataFrame
            Wide-format DataFrame with the full objective vector per
            generation.
        """

        return self.history_tracker.to_pandas_full_objective()
