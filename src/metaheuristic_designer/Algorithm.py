from __future__ import annotations
from typing import List, Tuple, Any
from abc import ABC, abstractmethod
import time
import json
import numpy as np
import pyparsing as pp
import matplotlib.pyplot as plt
from .utils import NumpyEncoder
from .ObjectiveFunc import ObjectiveFunc
from .SearchStrategy import SearchStrategy
from .ParamScheduler import ParamScheduler
from .Population import Population


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
    params: ParamScheduler or dict, optional
        Dictionary of parameters to define the stopping condition and output of the algorithm.
    name: str, optional
        Name that will be displayed when showing the algorithm.
    """

    def __init__(
        self,
        objfunc: ObjectiveFunc,
        search_strategy: SearchStrategy,
        params: ParamScheduler | dict = None,
        name: str = None,
    ):
        """
        Constructor of the Search class
        """

        self.params = params
        self.search_strategy = search_strategy
        self.objfunc = objfunc
        self._name = name

        if params is None:
            params = {}

        # Verbose parameters
        self.show_init_info = params.get("init_info", True)
        self.verbose = params.get("verbose", True)
        self.v_timer = params.get("v_timer", 1)

        # Stopping conditions
        self.stop_cond = params.get("stop_cond", "time_limit")
        self.stop_cond_parsed = parse_stopping_cond(self.stop_cond)

        self.progress_metric = params.get("progress_metric", self.stop_cond)
        self.progress_metric_parsed = parse_stopping_cond(self.progress_metric) if "progress_metric" in params else self.stop_cond_parsed

        self.ngen = params.get("ngen", 100)
        self.neval = params.get("neval", 1e5)
        self.time_limit = params.get("time_limit", 10.0)
        self.cpu_time_limit = params.get("cpu_time_limit", 10.0)
        self.fit_target = params.get("fit_target", 1e-10)
        if self.fit_target == 0:
            self.fit_target = 1e-10
        self.max_patience = params.get("patience", 1)
        self.patience_left = self.max_patience

        # Parallel parameters
        self.parallel = params.get("parallel", False)
        self.threads = params.get("threads", 8)

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

    @property
    def name(self):
        return self._name if self._name else self.search_strategy.name

    @name.setter
    def name(self, new_name: str):
        self._name = new_name

    @property
    def initializer(self):
        return self.search_strategy.initializer

    @initializer.setter
    def initializer(self, new_initializer):
        self.search_strategy.initializer = new_initializer

    def population(self) -> Population:
        return self.search_strategy.population

    def best_solution(self, decoded=False) -> Tuple[Any, float]:
        """
        Returns the best solution so far in the population.

        Returns
        -------
        best_solution: Tuple[Any, float]
            A pair of the best individual with its fitness.
        """

        return self.search_strategy.best_solution(decoded)

    def restart(self, reset_objfunc=True):
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
        if reset_objfunc:
            self.objfunc.counter = 0

    def save_solution(self, file_name: str = "solution.csv"):
        """
        Save the result of an execution to a csv file in disk.

        Parameters
        ----------

        file_name: str
            Path to the file where the solution will be stored.
        """

        ind, fit = self.search_strategy.best_solution(decoded=False)
        np.savetxt(file_name, ind.reshape([1, -1]), delimiter=",")

    def stopping_condition(self, gen: int, real_time_start: float, cpu_time_start: float) -> bool:
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

        neval_reached = self.objfunc.counter >= self.neval

        ngen_reached = gen >= self.ngen

        real_time_reached = time.time() - real_time_start >= self.time_limit

        cpu_time_reached = time.process_time() - cpu_time_start >= self.cpu_time_limit

        if self.objfunc.mode == "max":
            target_reached = self.best_solution()[1] >= self.fit_target
        else:
            target_reached = self.best_solution()[1] <= self.fit_target

        patience_reached = self.patience_left < 0

        return self.search_strategy.finish or process_condition(
            self.stop_cond_parsed,
            neval_reached,
            ngen_reached,
            real_time_reached,
            cpu_time_reached,
            target_reached,
            patience_reached,
        )

    def get_progress(self, gen: int, real_time_start: float, cpu_time_start: float) -> float:
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

        neval_reached = self.objfunc.counter / self.neval

        ngen_reached = gen / self.ngen

        real_time_reached = (time.time() - real_time_start) / self.time_limit

        cpu_time_reached = (time.process_time() - cpu_time_start) / self.cpu_time_limit

        best_fitness = self.best_solution()[1]

        fit_target = self.fit_target if self.fit_target != 0 else 1e-10
        if self.objfunc.mode == "max":
            target_reached = 1 - (best_fitness - self.fit_target) / fit_target
        else:
            target_reached = 1 - (self.fit_target - best_fitness) / fit_target

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

    def update(self, real_time_start: float, cpu_time_start: float, pass_step: bool = True):
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

        if self.prev_best_fitness is not None and ((self.best_solution()[1] >= self.prev_best_fitness) != (self.objfunc.mode == "max")):
            self.patience_left -= 1
        else:
            self.patience_left = self.max_patience

        self.prev_best_fitness = self.best_solution()[1]

        self.progress = self.get_progress(self.steps, real_time_start, cpu_time_start)

        self.ended = self.stopping_condition(self.steps, real_time_start, cpu_time_start)

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
    def step(self, time_start: float = 0, verbose: bool = False) -> Population:
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
        current_population: Population
            The new population obtained in this iteration of the algorithm.
        """

    def optimize(self, initialize=True) -> Population:
        """
        Execute the algorithm to get the best solution possible along with its evaluation.
        It will initialize the algorithm and repeat steps of the algorithm untill the
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
        display_timer = time.time()

        # Initizalize search strategy
        if initialize:
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

        return self.search_strategy.population

    def get_state(
        self,
        show_fit_history: bool = False,
        show_gen_history: bool = False,
        show_population: bool = False,
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

        data["search_strat_state"] = self.search_strategy.get_state(show_population)

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
            Save the best inividual for each iteration.
        show_pop: bool, optional
            Save the entire population of the last iteration.
        show_pop_details:bool, optional
            Save the detailed information of each individual.
        """

        dumped = json.dumps(
            self.get_state(
                show_fit_history,
                show_gen_history,
                show_population,
            ),
            cls=NumpyEncoder,
            indent=4 if readable else None,
        )

        with open(file_name, "w") as fp:
            fp.write(dumped)

    def init_info(self):
        print(f"Initializing optimization of {self.objfunc.name} using {self.search_strategy.name}")
        print(f"-----------------------------{'-'*len(self.objfunc.name)}-------{'-'*len(self.search_strategy.name)}")
        print()

    def step_info(self, start_time: float = 0):
        """
        Displays information about the current state of the algotithm.

        Parameters
        ----------
        time_start: float, optional
            Indicates to the algorihm how much time has already passed.
        """

        print(f"Optimizing {self.objfunc.name} using {self.name}:")
        print(f"\tReal time Spent: {round(time.time() - start_time,2)} s")
        print(f"\tCPU time Spent:  {round(time.time() - start_time,2)} s")
        print(f"\tGeneration: {self.steps}")
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
            fig, ax = plt.subplots()
            ax.plot(self.fit_history, color="blue", zorder=3)
            _xlim = ax.get_xlim()
            _ylim = ax.get_ylim()
            ax.axhline(y=0, color="black", alpha=0.9)
            ax.axvline(x=0, color="black", alpha=0.9)
            ax.set_xlim(_xlim)
            ax.set_ylim(_ylim)
            ax.set(xlabel="Generations", ylabel="Fitness", title=f"{self.search_strategy.name} fitness")
            ax.grid()
            plt.show()

        self.search_strategy.extra_report(show_plots)


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

    match cond_parsed:
        case [cond1, "and", cond2]:
            progress1 = process_progress(cond1, neval, ngen, real_time, cpu_time, target, patience)
            progress2 = process_progress(cond2, neval, ngen, real_time, cpu_time, target, patience)

            result = max(progress1, progress2)

        case [cond1, "or", cond2]:
            progress1 = process_progress(cond1, neval, ngen, real_time, cpu_time, target, patience)
            progress2 = process_progress(cond2, neval, ngen, real_time, cpu_time, target, patience)

            result = min(progress1, progress2)

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
