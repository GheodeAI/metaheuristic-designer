from typing import List, Optional
import logging
import time
from dataclasses import dataclass
import pyparsing as pp
import numpy as np
from .objective_function import ObjectiveFunc
from .population import Population

logger = logging.getLogger(__name__)


@dataclass
class StoppingCondition:
    condition_str: str = "time_limit"
    progress_metric_str: Optional[str] = None
    max_iterations: int = 1000
    max_evaluations: int = 1e5
    time_limit: float = 60.0
    cpu_time_limit: float = 60.0
    target_fitness: float = 1e-10
    max_patience: int = 100
    optimization_mode: str = "max"

    def __post_init__(self):
        self.stop_cond_parsed = parse_stopping_cond(self.condition_str)

        if self.progress_metric_str is None:
            self.progress_metric_str = self.condition_str
            self.progress_metric_parsed = self.stop_cond_parsed
        else:
            self.progress_metric_parsed = parse_stopping_cond(self.progress_metric_str)

        self.patience_left = self.max_patience
        self.iterations = 0
        self.evaluations = 0
        self.real_time_start = time.time()
        self.cpu_time_start = time.process_time()
        self.prev_best_fitness = None
        self.first_best_fitness = None

    def restart(self):
        self.iterations = 0
        self.evaluations = 0
        self.real_time_start = time.time()
        self.cpu_time_start = time.process_time()
        self.prev_best_fitness = None
        self.first_best_fitness = None

    def step(self, current_population: Population, skip_step=False):
        objfunc = current_population.objfunc
        _, best_fitness = current_population.best_solution()

        if not skip_step or self.optimization_mode not in {"max", "min"}:
            if self.prev_best_fitness is not None and ((best_fitness >= self.prev_best_fitness) != (self.optimization_mode == "max")):
                self.patience_left -= 1
            else:
                self.patience_left = self.max_patience

        if not skip_step:
            self.iterations += 1
        self.evaluations = objfunc.counter
        self.best_fitness = best_fitness
        self.real_time_spent = time.time() - self.real_time_start
        self.cpu_time_spent = time.time() - self.cpu_time_start
        if self.first_best_fitness is None:
            self.first_best_fitness = best_fitness
        self.prev_best_fitness = best_fitness

        logger.debug(
            "Updated stopping condition parameters:\nfunc. evaluations = %d\n"
            "generations = %d\ntime = %f\ncpu_time = %f\nbest = %f\npatience = %d",
            self.evaluations,
            self.iterations,
            self.real_time_start,
            self.cpu_time_start,
            self.best_fitness,
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

        neval_reached = self.evaluations >= self.max_evaluations
        ngen_reached = self.iterations >= self.max_iterations
        real_time_reached = self.real_time_spent >= self.time_limit
        cpu_time_reached = self.cpu_time_spent >= self.cpu_time_limit
        if self.optimization_mode == "max":
            target_reached = self.best_fitness >= self.target_fitness
        elif self.optimization_mode == "min":
            target_reached = self.best_fitness <= self.target_fitness
        patience_reached = self.patience_left <= 0

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
            self.best_fitness,
            self.target_fitness,
            self.patience_left,
        )

        return process_condition(
            self.stop_cond_parsed,
            neval_reached,
            ngen_reached,
            real_time_reached,
            cpu_time_reached,
            target_reached,
            patience_reached,
        )

    def get_progress(self) -> float:
        """
        Given the state of the algorithm, returns a number between 0 and 1 indicating
        how close to the end of the algorithm we are, 0 when starting and 1 when finished.

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
        progress: float
            Indicator of how close it the algorithm to finishing, 1 means the algorithm should be stopped.
        """

        tol = 1e-12
        neval_reached = self.evaluations / self.max_evaluations
        ngen_reached = self.iterations / self.max_iterations
        real_time_reached = self.real_time_spent / self.time_limit
        cpu_time_reached = self.cpu_time_spent / self.cpu_time_limit
        if self.first_best_fitness is not None and abs(self.first_best_fitness - self.target_fitness) <= tol * max(abs(self.first_best_fitness), abs(self.target_fitness), 1):
            target_progress = 1
        elif self.first_best_fitness is not None:
            fit_init_dist = self.first_best_fitness - self.target_fitness
            fit_dist = self.best_fitness - self.target_fitness
            target_progress = 1 - fit_dist / fit_init_dist
        else:
            target_progress = 0
        patience_percentage = 1 - self.patience_left / self.max_patience

        return process_progress(
            self.stop_cond_parsed,
            neval_reached,
            ngen_reached,
            real_time_reached,
            cpu_time_reached,
            target_progress,
            patience_percentage,
        )


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
    condition = pp.oneOf(["neval", "ngen", "time_limit", "cpu_time_limit", "fit_target", "convergence"])

    expr = pp.infixNotation(condition, [(orop, 2, pp.opAssoc.RIGHT), (andop, 2, pp.opAssoc.RIGHT)])

    return expr.parse_string(condition_str).as_list()


def process_condition(
    cond_parsed: List[str | List],
    neval: int,
    ngen: int,
    real_time: float,
    cpu_time: float,
    target: float,
    patience: int,
) -> bool:
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

        case "neval":
            result = neval

        case "ngen":
            result = ngen

        case "time_limit":
            result = real_time

        case "cpu_time_limit":
            result = cpu_time

        case "fit_target":
            result = target

        case "convergence":
            result = patience

    return result


def process_progress(
    cond_parsed: List[str | List],
    neval: int,
    ngen: int,
    real_time: float,
    cpu_time: float,
    target: float,
    patience: int,
) -> float:
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

        case "neval":
            result = neval

        case "ngen":
            result = ngen

        case "time_limit":
            result = real_time

        case "cpu_time_limit":
            result = cpu_time

        case "fit_target":
            result = target

        case "convergence":
            result = patience

    return result
