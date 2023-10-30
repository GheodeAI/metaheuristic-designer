from __future__ import annotations
from abc import ABC, abstractmethod
import time
import json
import numpy as np
import pyparsing as pp
from .utils import NumpyEncoder


class Algorithm(ABC):
    """
    General framework for metaheuristic algorithms.

    Parameters
    ----------

    objfunc: ObjectiveFunc
        Objective function to be optimized.
    search_strategy: SearchStrategy
        Search strategy that will iteratively optimize the function.
    params: ParamScheduler or dict, optional
        Dictionary of parameters to define the stopping condition and output of the algorithm.
    """

    def __init__(
        self,
        objfunc: ObjectiveFunc,
        search_strategy: SearchStrategy,
        params: Union[ParamScheduler, dict] = None,
    ):
        """
        Constructor of the Search class
        """

        self.params = params
        self.search_strategy = search_strategy
        self.objfunc = objfunc

        if params is None:
            params = {}

        # Verbose parameters
        self.verbose = params.get("verbose", True)
        self.v_timer = params.get("v_timer", 1)

        # Stopping conditions
        self.stop_cond = params.get("stop_cond", "time_limit")
        self.stop_cond_parsed = parse_stopping_cond(self.stop_cond)

        self.progress_metric = params.get("progress_metric", self.stop_cond)
        self.progress_metric_parsed = (
            parse_stopping_cond(self.progress_metric)
            if "progress_metric" in params
            else self.stop_cond_parsed
        )

        self.Ngen = params.get("ngen", 100)
        self.Neval = params.get("neval", 1e5)
        self.time_limit = params.get("time_limit", 10.0)
        self.cpu_time_limit = params.get("cpu_time_limit", 10.0)
        self.fit_target = params.get("fit_target", 1e-10)
        self.max_patience = params.get("patience", 1)
        self.patience_left = self.max_patience

        # Metrics
        self.fit_history = []
        self.best_history = []
        self.progress = 0
        self.ended = False
        self.steps = 0
        self.prev_best_fitness = None
        self.cpu_time_spent = 0
        self.real_time_spent = 0
        self.converged_steps = 0

    def restart(self):
        """
        Resets the internal values of the algorithm and the number of evaluations of the fitness function.
        """

        self.fit_history = []
        self.best_history = []
        self.progress = 0
        self.ended = False
        self.prev_best_fitness = None
        self.cpu_time_spent = 0
        self.real_time_spent = 0
        self.objfunc.counter = 0

    def save_solution(self, file_name: str = "solution.csv"):
        """
        Save the result of an execution to a csv file in disk.

        Parameters
        ----------

        file_name: str
            Path to the file where the solution will be stored.
        """

        ind, fit = self.search_strategy.best_solution()
        np.savetxt(file_name, ind.reshape([1, -1]), delimiter=",")

    def best_solution(self) -> Tuple[Individual, float]:
        """
        Returns the best solution so far in the population.

        Returns
        -------
        best_solution: Tuple[Individual, float]
            A pair of the best individual with its fitness.
        """

        return self.search_strategy.best_solution()

    def stopping_condition(
        self, gen: int, real_time_start: float, cpu_time_start: float
    ) -> bool:
        """
        Given the state of the algorithm, returns wether we have finished or not.

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
        has_stopped: bool
            Whether the algorithm has reached its end
        """

        neval_reached = self.objfunc.counter >= self.Neval

        ngen_reached = gen >= self.Ngen

        real_time_reached = time.time() - real_time_start >= self.time_limit

        cpu_time_reached = time.process_time() - cpu_time_start >= self.cpu_time_limit

        if self.objfunc.mode == "max":
            target_reached = self.best_solution()[1] >= self.fit_target
        else:
            target_reached = self.best_solution()[1] <= self.fit_target

        patience_reached = self.patience_left < 0

        return process_condition(
            self.stop_cond_parsed,
            neval_reached,
            ngen_reached,
            real_time_reached,
            cpu_time_reached,
            target_reached,
            patience_reached,
        )

    def get_progress(
        self, gen: int, real_time_start: float, cpu_time_start: float
    ) -> float:
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

        neval_reached = self.objfunc.counter / self.Neval

        ngen_reached = gen / self.Ngen

        real_time_reached = (time.time() - real_time_start) / self.time_limit

        cpu_time_reached = (time.process_time() - cpu_time_start) / self.cpu_time_limit

        best_fitness = self.best_solution()[1]
        if self.objfunc.mode == "max":
            target_reached = best_fitness / self.fit_target
        else:
            if best_fitness == 0:
                best_fitness = 1e-40
            target_reached = self.fit_target / best_fitness

        patience_prec = 1 - self.patience_left / self.max_patience

        return process_progress(
            self.stop_cond_parsed,
            neval_reached,
            ngen_reached,
            real_time_reached,
            cpu_time_reached,
            target_reached,
            patience_prec,
        )

    def update(
        self, real_time_start: float, cpu_time_start: float, pass_step: bool = True
    ):
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

        if pass_step:
            self.steps += 1

        if self.best_solution()[1] == self.prev_best_fitness:
            self.patience_left -= 1
        else:
            self.patience_left = self.max_patience

        self.prev_best_fitness = self.best_solution()[1]

        self.progress = self.get_progress(self.steps, real_time_start, cpu_time_start)

        self.ended = self.stopping_condition(
            self.steps, real_time_start, cpu_time_start
        )

    def initialize(self):
        """
        Initializes the optimization algorithm.
        """

        self.restart()
        self.search_strategy.initialize(self.objfunc)

    @abstractmethod
    def step(
        self, time_start: float = 0, verbose: bool = False
    ) -> Tuple[Individual, float]:
        """
        Performs an iteration of the algorithm.

        Parameters
        ----------
        time_start: float, optional
            Indicates to the algorihm how much time has already passed.
        verbose: bool, optional
            Indicates whether to show the status of the algorithm or not.

        Returns
        -------
        best_solution: Tuple[Individual, float]
            A pair of the best individual with its fitness.
        """

    def optimize(self) -> Tuple[Individual, float]:
        """
        Execute the algorithm to get the best solution possible along with its evaluation.
        It will initialize the algorithm and repeat steps of the algorithm untill the
        stopping condition is met.

        Returns
        -------
        best_solution: Tuple[Individual, float]
            A pair of the best individual with its fitness.
        """

        if self.verbose:
            self.init_info()

        self.steps = 0

        # initialize clocks
        real_time_start = time.time()
        cpu_time_start = time.process_time()
        display_timer = time.time()

        # Initizalize search strategy
        self.initialize()

        # Search until the stopping condition is met
        self.update(real_time_start, cpu_time_start, pass_step=False)

        if self.verbose:
            self.step_info(real_time_start)

        while not self.ended:
            self.step(real_time_start)

            self.update(real_time_start, cpu_time_start)

            # Display information
            if self.verbose and time.time() - display_timer > self.v_timer:
                self.step_info(real_time_start)
                display_timer = time.time()

        # Store the time spent optimizing
        self.real_time_spent = time.time() - real_time_start
        self.cpu_time_spent = time.process_time() - cpu_time_start

        return self.best_solution()

    def get_state(
        self,
        show_best_solution: bool = True,
        show_fit_history: bool = False,
        show_gen_history: bool = False,
        show_pop: bool = False,
        show_pop_details: bool = False,
    ) -> dict:
        """
        Gets the current state of the algorithm as a dictionary.

        Parameters
        ----------
        show_best_solution: bool, optional
            Save the best solution found by the algorithm.
        show_fit_history: bool, optional
            Save the fitness of the best individual of each iteration.
        show_gen_history: bool, optional
            Save the best inividual for each iteration.
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
            "ended": self.ended,
            "progress": self.progress,
            "generation": self.steps,
            "evaluations": self.objfunc.counter,
            "real_time_spent": self.real_time_spent,
            "cpu_time_spent": self.cpu_time_spent,
            "params": self.params,
        }

        if show_best_solution:
            data["best_fitness"] = self.best_solution()[1]
            data["best_individual"] = self.search_strategy.best.get_state(
                show_speed=False, show_best=False
            )

        if show_fit_history:
            data["fit_history"] = self.fit_history

        if show_gen_history:
            data["best_history"] = self.best_history

        data["search_strat_state"] = self.search_strategy.get_state(
            show_pop, show_pop_details
        )

        return data

    def store_state(
        self,
        file_name: str = "dumped_state.json",
        readable: bool = False,
        show_best_solution: bool = True,
        show_fit_history: bool = False,
        show_gen_history: bool = False,
        show_pop: bool = False,
        show_pop_details: bool = False,
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
            Save the best inividual for each iteration.
        show_pop: bool, optional
            Save the entire population of the last iteration.
        show_pop_details:bool, optional
            Save the detailed information of each individual.
        """

        dumped = json.dumps(
            self.get_state(
                show_best_solution,
                show_fit_history,
                show_gen_history,
                show_pop,
                show_pop_details,
            ),
            cls=NumpyEncoder,
            indent=4 if readable else None,
        )

        with open(file_name, "w") as fp:
            fp.write(dumped)

    def init_info(self):
        print(
            f"Initializing optimization of {self.objfunc.name} using {self.search_strategy.name}"
        )
        print(
            f"-----------------------------{'-'*len(self.objfunc.name)}-------{'-'*len(self.search_strategy.name)}"
        )
        print()

    @abstractmethod
    def step_info(self, start_time: float = 0):
        """
        Displays information about the current state of the algotithm.

        Parameters
        ----------
        time_start: float, optional
            Indicates to the algorihm how much time has already passed.
        """

    @abstractmethod
    def display_report(self, show_plots: bool = True):
        """
        Shows a summary of the execution of the algorithm.

        Parameters
        ----------
        show_plots: bool, optional
            Whether to display plots about the algorithm or not.
        """


def parse_stopping_cond(condition_str: str) -> List[str | List]:
    """
    This function parses an expression of the form "neval or cpu_time" into
    a tree structure so that it can be futher processed.

    Parameters
    ----------
    condition_str: str
        The string to be parsed.

    Returns
    -------
    token_list: List[str | List]
        The list of tokens representing the original string.
    """

    orop = pp.Literal("and")
    andop = pp.Literal("or")
    condition = pp.oneOf(
        ["neval", "ngen", "time_limit", "cpu_time_limit", "fit_target", "convergence"]
    )

    expr = pp.infixNotation(
        condition, [(orop, 2, pp.opAssoc.RIGHT), (andop, 2, pp.opAssoc.RIGHT)]
    )

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
    This function recieves as an input an expression for the stopping condition
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

    if isinstance(cond_parsed, list):
        if len(cond_parsed) == 3:
            cond1 = process_condition(
                cond_parsed[0], neval, ngen, real_time, cpu_time, target, patience
            )
            cond2 = process_condition(
                cond_parsed[2], neval, ngen, real_time, cpu_time, target, patience
            )

            if cond_parsed[1] == "or":
                result = cond1 or cond2
            elif cond_parsed[1] == "and":
                result = cond1 and cond2

        elif len(cond_parsed) == 1:
            result = process_condition(
                cond_parsed[0], neval, ngen, real_time, cpu_time, target, patience
            )

    else:
        if cond_parsed == "neval":
            result = neval
        elif cond_parsed == "ngen":
            result = ngen
        elif cond_parsed == "time_limit":
            result = real_time
        elif cond_parsed == "cpu_time_limit":
            result = cpu_time
        elif cond_parsed == "fit_target":
            result = target
        elif cond_parsed == "convergence":
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
    This function recieves as an input an expression for the stopping condition
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

    if isinstance(cond_parsed, list):
        if len(cond_parsed) == 3:
            progress1 = process_progress(
                cond_parsed[0], neval, ngen, real_time, cpu_time, target, patience
            )
            progress2 = process_progress(
                cond_parsed[2], neval, ngen, real_time, cpu_time, target, patience
            )

            if cond_parsed[1] == "or":
                result = max(progress1, progress2)
            elif cond_parsed[1] == "and":
                result = min(progress1, progress2)

        elif len(cond_parsed) == 1:
            result = process_progress(
                cond_parsed[0], neval, ngen, real_time, cpu_time, target, patience
            )
    else:
        if cond_parsed == "neval":
            result = neval
        elif cond_parsed == "ngen":
            result = ngen
        elif cond_parsed == "time_limit":
            result = real_time
        elif cond_parsed == "cpu_time_limit":
            result = cpu_time
        elif cond_parsed == "fit_target":
            result = target
        elif cond_parsed == "convergence":
            result = patience

    return result
