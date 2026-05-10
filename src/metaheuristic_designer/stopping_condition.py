"""
Module for algorithm stopping conditions and progress metric evaluation.
"""

from typing import List, Optional
import logging
import time
from dataclasses import dataclass
import pyparsing as pp
from .population import Population

logger = logging.getLogger(__name__)


@dataclass
class StoppingCondition:
    """Encapsulate the logic that decides when an optimisation run should end.

    A stopping condition is built from a logical expression that
    combines **tokens** with ``and``, ``or`` and parentheses.  Each
    token has a corresponding numeric limit.  The same expression
    (or a separate one) can be used to compute a progress value
    between 0 and 1 for parameter schedules.

    Parameters
    ----------
    condition_str : str
        Logical expression defining when to stop (e.g.
        ``"max_iterations or real_time_limit"``).
    progress_metric_str : str, optional
        Logical expression defining how to compute the 0-1 progress
        value.  Defaults to *condition_str*.
    max_iterations : int, optional
        Maximum number of generations.
    max_evaluations : int, optional
        Maximum number of objective function evaluations.
    real_time_limit : float, optional
        Wall-clock time limit in seconds.
    cpu_time_limit : float, optional
        CPU time limit in seconds.
    objective_target : float, optional
        Target value for the raw objective.
    max_patience : int, optional
        Consecutive iterations without improvement before
        ``"convergence"`` triggers.
    optimization_mode : str, optional
        ``"max"`` or ``"min"``, how the objective target and
        convergence are evaluated.
    """

    condition_str: str
    progress_metric_str: Optional[str] = None
    max_iterations: int = None
    max_evaluations: int = None
    real_time_limit: float = None
    cpu_time_limit: float = None
    objective_target: float = None
    max_patience: int = None
    optimization_mode: str = "max"

    def __post_init__(self):
        self.condition_str = self.condition_str
        self.stop_cond_parsed = parse_stopping_cond(self.condition_str)
        self._validate_required_params(self.condition_str, "stopping condition")

        if self.progress_metric_str is None:
            self.progress_metric_str = self.condition_str
            self.progress_metric_parsed = self.stop_cond_parsed
        else:
            self.progress_metric_parsed = parse_stopping_cond(self.progress_metric_str)
            self._validate_required_params(self.progress_metric_str, "progress metric")

        # Set max patience to a sensible value, we need it to keep track of the missed iterations
        if self.max_patience is None:
            self.max_patience = 1
        self.patience_left = self.max_patience
        self.iterations = 0
        self.evaluations = 0
        self.real_time_start = time.time()
        self.cpu_time_start = time.process_time()
        self.real_time_spent = 0
        self.cpu_time_spent = 0
        self.prev_best_objective = None
        self.first_best_objective = None
        self.best_objective = None

    def _validate_required_params(self, source_str: str, context: str):
        """Raise ValueError if a token appears in *source_str* but the
        corresponding parameter is None."""
        # Mapping from token to (attribute_name, value)
        _token_map = {
            "max_evaluations": ("max_evaluations", self.max_evaluations),
            "max_iterations": ("max_iterations", self.max_iterations),
            "real_time_limit": ("real_time_limit", self.real_time_limit),
            "cpu_time_limit": ("cpu_time_limit", self.cpu_time_limit),
            "objective_target": ("objective_target", self.objective_target),
            "convergence": ("max_patience", self.max_patience),
        }

        for token, (attr, attr_value) in _token_map.items():
            if token in source_str and attr_value is None:
                raise ValueError(f'"{token}" appears in the {context} but "{attr}" is not set.')

    def restart(self):
        """Reset all counters and timers for a fresh run."""
        self.iterations = 0
        self.evaluations = 0
        self.real_time_start = time.time()
        self.cpu_time_start = time.process_time()
        self.real_time_spent = 0
        self.cpu_time_spent = 0
        self.prev_best_objective = None
        self.first_best_objective = None

    def step(self, current_population: Population):
        """Advance internal counters after one generation.

        Parameters
        ----------
        current_population : Population
            The population at the end of the current generation.  Its
            best objective is used to update convergence tracking.
        """

        objfunc = current_population.objfunc

        self.iterations += 1
        self.evaluations = objfunc.counter
        self.real_time_spent = time.time() - self.real_time_start
        self.cpu_time_spent = time.process_time() - self.cpu_time_start

        if self.optimization_mode in {"max", "min"}:
            _, best_objective = current_population.best_solution()
            if self.prev_best_objective is not None:
                if self.optimization_mode == "max":
                    improves = best_objective <= self.prev_best_objective
                else:
                    improves = best_objective >= self.prev_best_objective

                if improves:
                    self.patience_left -= 1
                else:
                    self.patience_left = self.max_patience
                    if self.first_best_objective is None:
                        self.first_best_objective = best_objective
                    self.prev_best_objective = best_objective
                    self.best_objective = best_objective
            else:
                self.prev_best_objective = best_objective

        logger.debug(
            "Updated stopping condition parameters:\nfunc. evaluations = %d\n" "generations = %d\ntime = %f\ncpu_time = %f\nbest = %f\npatience = %d",
            self.evaluations,
            self.iterations,
            self.real_time_start,
            self.cpu_time_start,
            self.best_objective,
            self.patience_left,
        )

    def is_finished(self, finished: bool = False) -> bool:
        """
        Given the state of the algorithm, returns wether we have finished or not.

        Parameters
        ----------
        real_time_start: float
            The time in seconds that passed since the algorithm was executed.
        cpu_time_start: float
            The time in seconds that the CPU has executed code in this algorithm.

        Returns
        -------
        has_stopped: bool
            Whether the algorithm has reached its end
        """

        if finished:
            return True

        neval_reached = self.evaluations >= self.max_evaluations if self.max_evaluations is not None else False
        ngen_reached = self.iterations >= self.max_iterations if self.max_iterations is not None else False
        real_time_reached = self.real_time_spent >= self.real_time_limit if self.real_time_limit is not None else False
        cpu_time_reached = self.cpu_time_spent >= self.cpu_time_limit if self.cpu_time_limit is not None else False

        if (self.objective_target is not None) and (self.optimization_mode == "max"):
            target_reached = (self.best_objective is not None) and (self.best_objective >= self.objective_target)
        elif (self.objective_target is not None) and (self.optimization_mode == "min"):
            target_reached = (self.best_objective is not None) and (self.best_objective <= self.objective_target)
        else:
            target_reached = False

        if (self.patience_left is not None) and (self.optimization_mode in {"min", "max"}):
            patience_reached = self.patience_left <= 0
        else:
            patience_reached = False

        logger.debug(
            "Evaluating stopping condition:\n\tcondition: %s\n\tfunc. evaluations = %d / %d"
            "\n\tgenerations = %d / %d\n\ttime = %f / %f\n\tcpu_time = %f / %f\n\tbest = %f / %f\n\tpatience = %d",
            self.condition_str,
            self.max_evaluations,
            self.evaluations,
            self.iterations,
            self.max_iterations,
            self.real_time_start,
            time.time(),
            self.cpu_time_start,
            time.process_time(),
            self.best_objective,
            self.objective_target,
            self.patience_left,
        )

        return process_condition(
            self.stop_cond_parsed, neval_reached, ngen_reached, real_time_reached, cpu_time_reached, target_reached, patience_reached
        )

    def get_progress(self) -> float:
        """
        Compute the current progress (0-1) according to the progress metric.
        

        Parameters
        ----------
        gen: int
            The number of generations that has passed.
        real_time_start: float
            The time in seconds that passed since the algorithm was executed.
        cpu_time_start: float
            The time in seconds that the CPU has executed code in this algorithm.

        Returns
        -------
        float
            A value between 0 (start) and 1 (finished).
        """

        tol = 1e-12
        neval_reached = self.evaluations / self.max_evaluations if self.max_evaluations is not None else 0
        ngen_reached = self.iterations / self.max_iterations if self.max_iterations is not None else 0
        real_time_reached = self.real_time_spent / self.real_time_limit if self.real_time_limit is not None else 0
        cpu_time_reached = self.cpu_time_spent / self.cpu_time_limit if self.cpu_time_limit is not None else 0

        if (self.objective_target is not None) and (self.optimization_mode in {"min", "max"}):
            if self.first_best_objective is not None and abs(self.first_best_objective - self.objective_target) <= tol * max(
                abs(self.first_best_objective), abs(self.objective_target), 1
            ):
                target_progress = 1
            elif self.first_best_objective is not None:
                fit_init_dist = self.first_best_objective - self.objective_target
                fit_dist = self.best_objective - self.objective_target
                target_progress = 1 - fit_dist / fit_init_dist
            else:
                target_progress = 0
        else:
            target_progress = 0

        if (self.max_patience is not None) and (self.optimization_mode in {"min", "max"}):
            patience_percentage = 1 - self.patience_left / self.max_patience
        else:
            patience_percentage = 0

        return process_progress(
            self.progress_metric_parsed, neval_reached, ngen_reached, real_time_reached, cpu_time_reached, target_progress, patience_percentage
        )

    def get_state(self):
        """Return a dictionary with the current state of the stopping condition.

        Returns
        -------
        dict
            Keys include ``iterations``, ``evaluations``, times, and
            configuration limits.
        """

        data = {
            "class_name": self.__class__.__name__,
            "stopped": self.is_finished(),
            "progress": self.get_progress(),
            "stop_condition": self.condition_str,
            "progress_metric": self.progress_metric_str,
            "max_patience": self.max_patience,
            "max_iterations": self.max_iterations,
            "max_evaluations": self.max_evaluations,
            "real_time_limit": self.real_time_limit,
            "cpu_time_limit": self.cpu_time_limit,
            "patience_left": self.patience_left,
            "iterations": self.iterations,
            "evaluations": self.evaluations,
            "real_time_spent": self.real_time_spent,
            "cpu_time_spent": self.cpu_time_spent,
        }

        return data


def parse_stopping_cond(condition_str: str) -> List[str | List]:
    """
    This function parses an expression of the form "neval or cpu_time" into
    a tree structure so that it can be further processed.

    Parameters
    ----------
    condition_str: str
        The string to be parsed.

    Returns
    -------
    token_list: List[str | List]
        The list of tokens representing the original string.
    """

    orop = pp.Literal("or")
    andop = pp.Literal("and")
    condition = pp.one_of(["max_evaluations", "max_iterations", "real_time_limit", "cpu_time_limit", "objective_target", "convergence"])

    expr = pp.infix_notation(condition, [(orop, 2, pp.opAssoc.RIGHT), (andop, 2, pp.opAssoc.RIGHT)])

    return expr.parse_string(condition_str).as_list()


def process_condition(cond_parsed: List[str | List], neval: int, ngen: int, real_time: float, cpu_time: float, target: float, patience: int) -> bool:
    """
    This function receives as an input an expression for the stopping condition
    and the truth variable of the possible stopping conditions and returns wether to stop or not.

    Parameters
    ----------
    cond_parsed: List[str | List]
        The list of tokens representing the parsed stopping condition.
    neval: int
        Number of function evaluations done.
    ngen: int
        Number of iterations done by the algorithm
    real_time: float
        Time since the start of the algorithm.
    cpu_time: float
        Time dedicated by the CPU to optimizing our function.
    target: float
        Fitness target.
    patience: int
        Number of time the algorithm has reached the same fitness value in a row.

    Returns
    -------
    has_stopped: bool
        Whether the algorithm has reached its end
    """

    result = None

    match cond_parsed:
        case [cond1, "and", cond2]:
            cond1_parsed = process_condition(cond1, neval, ngen, real_time, cpu_time, target, patience)
            cond2_parsed = process_condition(cond2, neval, ngen, real_time, cpu_time, target, patience)

            result = cond1_parsed and cond2_parsed

        case [cond1, "or", cond2]:
            cond1_parsed = process_condition(cond1, neval, ngen, real_time, cpu_time, target, patience)
            cond2_parsed = process_condition(cond2, neval, ngen, real_time, cpu_time, target, patience)

            result = cond1_parsed or cond2_parsed

        case [cond1]:
            result = process_condition(cond1, neval, ngen, real_time, cpu_time, target, patience)

        case "max_evaluations":
            result = neval

        case "max_iterations":
            result = ngen

        case "real_time_limit":
            result = real_time

        case "cpu_time_limit":
            result = cpu_time

        case "objective_target":
            result = target

        case "convergence":
            result = patience

    return result


def process_progress(cond_parsed: List[str | List], neval: int, ngen: int, real_time: float, cpu_time: float, target: float, patience: int) -> float:
    """
    This function receives as an input an expression for the stopping condition
    and the truth variable of the possible stopping conditions and returns wether to stop or not.

    Parameters
    ----------
    cond_parsed: List[str | List]
        The list of tokens representing the parsed stopping condition.
    neval: int
        Number of function evaluations done.
    ngen: int
        Number of iterations done by the algorithm
    real_time: float
        Time since the start of the algorithm.
    cpu_time: float
        Time dedicated by the CPU to optimizing our function.
    target: float
        Fitness target.
    patience: int
        Number of time the algorithm has reached the same fitness value in a row.

    Returns
    -------
    has_stopped: bool
        Indicator of how close it the algorithm to finishing, 1 means the algorithm should be stopped.
    """

    result = None

    match cond_parsed:
        case [cond1, "and", cond2]:
            progress1 = process_progress(cond1, neval, ngen, real_time, cpu_time, target, patience)
            progress2 = process_progress(cond2, neval, ngen, real_time, cpu_time, target, patience)

            result = min(progress1, progress2)

        case [cond1, "or", cond2]:
            progress1 = process_progress(cond1, neval, ngen, real_time, cpu_time, target, patience)
            progress2 = process_progress(cond2, neval, ngen, real_time, cpu_time, target, patience)

            result = max(progress1, progress2)

        case [cond1]:
            result = process_progress(cond1, neval, ngen, real_time, cpu_time, target, patience)

        case "max_evaluations":
            result = neval

        case "max_iterations":
            result = ngen

        case "real_time_limit":
            result = real_time

        case "cpu_time_limit":
            result = cpu_time

        case "objective_target":
            result = target

        case "convergence":
            result = patience

    return result
