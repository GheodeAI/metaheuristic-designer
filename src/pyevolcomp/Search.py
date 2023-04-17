from __future__ import annotations
from abc import ABC, abstractmethod
import time
import numpy as np
import pyparsing as pp
from .ObjectiveFunc import ObjectiveVectorFunc
from .Initializers import UniformVectorInitializer

import json

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

        self.progress_metric = params["progress_metric"] if "progress_metric" in params else self.stop_cond
        self.progress_metric_parsed = parse_stopping_cond(self.progress_metric) if "progress_metric" in params else self.stop_cond_parsed

        self.Ngen = params["ngen"] if "ngen" in params else 100
        self.Neval = params["neval"] if "neval" in params else 1e5
        self.time_limit = params["time_limit"] if "time_limit" in params else 10.0
        self.cpu_time_limit = params["cpu_time_limit"] if "cpu_time_limit" in params else 10.0
        self.fit_target = params["fit_target"] if "fit_target" in params else 0

        # Metrics
        self.fit_history = []
        self.best_history = []
        self.progress = 0
        self.ended = False
        self.steps = 0
        self.best_fitness = 0
        self.cpu_time_spent = 0
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
        self.cpu_time_spent = 0
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
    

    def stopping_condition(self, gen, real_time_start, cpu_time_start):
        """
        Given the state of the algorithm, returns wether we have finished or not.
        """

        neval_reached = self.objfunc.counter >= self.Neval

        ngen_reached = gen >= self.Ngen

        real_time_reached = time.time() - real_time_start >= self.time_limit

        cpu_time_reached = time.process_time() - cpu_time_start >= self.cpu_time_limit

        if self.objfunc.mode == "max":
            target_reached = self.best_solution()[1] >= self.fit_target
        else:
            target_reached = self.best_solution()[1] <= self.fit_target

        return process_condition(self.stop_cond_parsed, neval_reached, ngen_reached, real_time_reached, cpu_time_reached, target_reached)

    def get_progress(self, gen, real_time_start, cpu_time_start):
        """
        Given the state of the algorithm, returns a number between 0 and 1 indicating
        how close to the end of the algorithm we are, 0 when starting and 1 when finished.
        """

        neval_reached = self.objfunc.counter/self.Neval
        
        ngen_reached = gen/self.Ngen

        real_time_reached = (time.time() - real_time_start)/self.time_limit

        cpu_time_reached = (time.process_time() - cpu_time_start)/self.cpu_time_limit

        best_fitness = self.best_solution()[1]
        if self.objfunc.mode == "max":
            target_reached = best_fitness/self.fit_target
        else:
            if best_fitness == 0:
                best_fitness = 1e-40
            target_reached = self.fit_target/best_fitness

        return process_progress(self.stop_cond_parsed, neval_reached, ngen_reached, real_time_reached, cpu_time_reached, target_reached)     

    def update(self, real_time_start, cpu_time_start, pass_step=True):
        """
        Given the state of the algorithm, returns a number between 0 and 1 indicating
        how close to the end of the algorithm we are, 0 when starting and 1 when finished.
        """        

        if pass_step:
            self.steps += 1

        self.progress = self.get_progress(self.steps, real_time_start, cpu_time_start)

        self.ended = self.stopping_condition(self.steps, real_time_start, cpu_time_start)

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

    def optimize(self) -> Tuple[Individual, float]:
        """
        Execute the algorithm to get the best solution possible along with it's evaluation
        """

        self.steps = 0

        # initialize clocks
        real_time_start = time.time()
        cpu_time_start = time.process_time()
        display_timer = time.time()

        # Initizalize search strategy
        self.initialize()

        # Search untill the stopping condition is met
        self.update(real_time_start, cpu_time_start, pass_step=False)
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

    def get_state(self, best_solution=True, fit_history=False, genotype_history=False, population=False):
        """
        Gets the current state of the algorithm as a dictionary
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

        if best_solution:
            data["best_individual"] = self.search_strategy.best.get_state()

        if fit_history:
            data["fit_history"] = self.fit_history

        if genotype_history:
            data["best_history"] = self.best_history

        data["search_strat_state"] = self.search_strategy.get_state(population=False)


        return data
    
    def store_state(self, file_name="dumped_state.json", readable=False):
        """
        Dumps the current state of the algorithm to a file.

        Everything will be stored in a JSON file.
        """

        dumped = json.dumps(self.get_state(), cls=NumpyEncoder, indent = 4 if readable else None)

        with open(file_name, "w") as fp:
            fp.write(dumped)
    
    
    def load_state(self):
        """
        Loads the state of the algorithm from a file
        """
    
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


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

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


def process_condition(cond_parsed, neval, ngen, real_time, cpu_time, target):
    """
    This function recieves as an input an expression for the stopping condition
    and the truth variable of the possible stopping conditions and returns wether to stop or not.
    """
    result = None

    if isinstance(cond_parsed, list):
        if len(cond_parsed) == 3:
            cond1 = process_condition(cond_parsed[0], neval, ngen, real_time, cpu_time, target)
            cond2 = process_condition(cond_parsed[2], neval, ngen, real_time, cpu_time, target)

            if cond_parsed[1] == "or":
                result = cond1 or cond2
            elif cond_parsed[1] == "and":
                result = cond1 and cond2

        elif len(cond_parsed) == 1:
            result = process_condition(cond_parsed[0], neval, ngen, real_time, cpu_time, target)

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
    
    return result

def process_progress(cond_parsed, neval, ngen, real_time, cpu_time, target):
    """
    This function recieves as an input an expression for the stopping condition 
    and the truth variable of the possible stopping conditions and returns wether to stop or not. 
    """
    result = None
    
    if isinstance(cond_parsed, list):
        if len(cond_parsed) == 3:
            
            progress1 = process_progress(cond_parsed[0], neval, ngen, real_time, cpu_time, target)
            progress2 = process_progress(cond_parsed[2], neval, ngen, real_time, cpu_time, target)

            if cond_parsed[1] == "or":
                result = max(progress1, progress2)
            elif cond_parsed[1] == "and":
                result = min(progress1, progress2)
            
        elif len(cond_parsed) == 1:
            result = process_progress(cond_parsed[0], neval, ngen, real_time, cpu_time, target)
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

    return result
