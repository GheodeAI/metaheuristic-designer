from __future__ import annotations
from abc import ABC, abstractmethod
import time
import numpy as np
import pyparsing as pp
from .ObjectiveFunc import ObjectiveVectorFunc
from .Initializers import UniformVectorInitializer


class Search(ABC):
    """
    General framework for metaheuristic algorithms
    """

    def __init__(self, objfunc: ObjectiveFunc, search_strategy: Algorithm, pop_init: Initializer = None, params: Union[ParamScheduler, dict] = None):
        """
        Constructor of the Metaheuristic class
        """

        self.params = params
        self.search_strategy = search_strategy
        self.objfunc = objfunc

        self.pop_init = pop_init
        
        if pop_init is None:
            if not isinstance(objfunc, ObjectiveVectorFunc):
                raise ValueError("A population initializer must be indicated.")
            else:
                self.pop_init = UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, search_strategy.popsize)
        else:
            self.pop_init.popSize = search_strategy.popsize

        if params is None:
            params = {}

        # Verbose parameters
        self.verbose = params["verbose"] if "verbose" in params else True
        self.v_timer = params["v_timer"] if "v_timer" in params else 1

        # Stopping conditions
        self.stop_cond = params["stop_cond"] if "stop_cond" in params else "time_limit"
        self.stop_cond_parsed = parse_stopping_cond(self.stop_cond)
        self.Ngen = params["ngen"] if "ngen" in params else 100
        self.Neval = params["neval"] if "neval" in params else 1e5
        self.time_limit = params["time_limit"] if "time_limit" in params else 10.0
        self.fit_target = params["fit_target"] if "fit_target" in params else 0

        # Metrics
        self.fit_history = []
        self.best_history = []
        self.progress = 0
        self.ended = False
        self.steps = 0
        self.best_fitness = 0
        self.time_spent = 0
        self.real_time_spent = 0

    def restart(self):
        """
        Resets the internal values of the algorithm and the number of evaluations of the fitness function.
        """

        self.fit_history = []
        self.best_history = []
        self.progress = 0
        self.ended = False
        self.best_fitness = 0
        self.time_spent = 0
        self.real_time_spent = 0

    def save_solution(self, file_name="solution.csv"):
        """
        Save the result of an execution to a csv file in disk.
        """

        ind, fit = self.search_strategy.best_solution()
        np.savetxt(file_name, ind.reshape([1, -1]), delimiter=',')
        with open(file_name, "a") as file:
            file.write(str(fit))

    def best_solution(self):
        """
        Returns the best solution so far in the population.
        """

        return self.search_strategy.best_solution()

    def stopping_condition(self, gen, real_time_start):
        """
        Given the state of the algorithm, returns wether we have finished or not.
        """

        neval_reached = self.objfunc.counter >= self.Neval

        ngen_reached = gen >= self.Ngen

        time_reached = time.time() - real_time_start >= self.time_limit

        if self.objfunc.mode == "max":
            target_reached = self.best_solution()[1] >= self.fit_target
        else:
            target_reached = self.best_solution()[1] <= self.fit_target

        return process_condition(self.stop_cond_parsed, neval_reached, ngen_reached, time_reached, target_reached)

    def get_progress(self, gen, time_start):
        """
        Given the state of the algorithm, returns a number between 0 and 1 indicating
        how close to the end of the algorithm we are, 0 when starting and 1 when finished.
        """

        prog = 0
        if self.stop_cond == "neval":
            prog = self.objfunc.counter / self.Neval
        elif self.stop_cond == "ngen":
            prog = gen / self.Ngen
        elif self.stop_cond == "time_limit":
            prog = (time.time() - time_start) / self.time_limit
        elif self.stop_cond == "fit_target":
            best_fitness = self.best_solution()[1]
            if self.objfunc.mode == "max":
                prog = best_fitness / self.fit_target
            else:
                if best_fitness == 0:
                    best_fitness = 1e-40
                prog = self.fit_target / best_fitness

        return prog

    def update(self, gen, time_start):
        """
        Given the state of the algorithm, returns a number between 0 and 1 indicating
        how close to the end of the algorithm we are, 0 when starting and 1 when finished.
        """

        self.progress = self.get_progress(gen, time_start)

        self.ended = self.stopping_condition(gen, time_start)

    def initialize(self):
        """
        Generates a random population of individuals
        """

        self.restart()
        initial_population = self.pop_init.generate_population(self.objfunc)
        self.search_strategy.initialize(initial_population)

    @abstractmethod
    def step(self, time_start=0, verbose=False) -> Tuple[Individual, float]:
        """
        Performs a step in the algorithm
        """

    def optimize(self):
        """
        Execute the algorithm to get the best solution possible along with it's evaluation
        """

        self.steps = 0

        # initialize clocks
        time_start = time.process_time()
        real_time_start = time.time()
        display_timer = time.time()

        # Initizalize search strategy
        self.initialize()

        # Search untill the stopping condition is met
        self.update(self.steps, real_time_start)
        while not self.ended:

            self.step(real_time_start)

            # Display information
            if self.verbose and time.time() - display_timer > self.v_timer:
                self.step_info(real_time_start)
                display_timer = time.time()

        # Store the time spent optimizing
        self.real_time_spent = time.time() - real_time_start
        self.time_spent = time.process_time() - time_start

        return self.best_solution()

    @abstractmethod
    def step_info(self, start_time):
        """
        Displays information about the current state of the algotithm
        """

    @abstractmethod
    def display_report(self, show_plots=True):
        """
        Shows a summary of the execution of the algorithm
        """


# Stopping condition string parsing methods
def parse_stopping_cond(condition_str):
    """
    This function parses an expression of the form "neval or cpu_time" into
    a tree structure so that it can be futher processed.
    """

    orop = pp.Literal("and")
    andop = pp.Literal("or")
    condition = pp.oneOf(["neval", "ngen", "time_limit", "cpu_time_limit", "fit_target"])

    expr = pp.infixNotation(
        condition,
        [
            (orop, 2, pp.opAssoc.RIGHT),
            (andop, 2, pp.opAssoc.RIGHT)
        ]
    )

    return expr.parse_string(condition_str).as_list()


def process_condition(cond_parsed, neval, ngen, real_time, target):
    """
    This function recieves as an input an expression for the stopping condition
    and the truth variable of the possible stopping conditions and returns wether to stop or not.
    """
    result = None

    if isinstance(cond_parsed, list):
        if len(cond_parsed) == 3:
            cond1 = process_condition(cond_parsed[0], neval, ngen, real_time, target)
            cond2 = process_condition(cond_parsed[2], neval, ngen, real_time, target)

            if cond_parsed[1] == "or":
                result = cond1 or cond2
            elif cond_parsed[1] == "and":
                result = cond1 and cond2

        elif len(cond_parsed) == 1:
            result = process_condition(cond_parsed[0], neval, ngen, real_time, target)
    else:
        if cond_parsed == "neval":
            result = neval
        elif cond_parsed == "ngen":
            result = ngen
        elif cond_parsed == "time_limit":
            result = real_time
        elif cond_parsed == "fit_target":
            result = target

    return result
