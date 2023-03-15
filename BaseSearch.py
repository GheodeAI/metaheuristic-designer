from __future__ import annotations
from typing import Tuple, List
from typing import Union
from .ParamScheduler import ParamScheduler
from .BaseAlgorithm import BaseAlgorithm
from abc import ABC, abstractmethod
import time
import numpy as np
import matplotlib.pyplot as plt


class BaseSearch(ABC):
    """
    General framework for metaheuristic algorithms
    """
    
    def __init__(self, search_strategy: BaseAlgorithm, params: Union[ParamScheduler, dict]):
        """
        Constructor of the Metaheuristic class
        """

        self.params = params
        self.search_strategy = search_strategy

        # Verbose parameters
        self.verbose = params["verbose"] if "verbose" in params else True
        self.v_timer = params["v_timer"] if "v_timer" in params else 1

        # Stopping conditions
        self.stop_cond = params["stop_cond"] if "stop_cond" in params else "time_limit"
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
    

    def stopping_condition(self, gen, time_start, objfunc):
        """
        Given the state of the algorithm, returns wether we have finished or not.
        """

        stop = True
        if self.stop_cond == "neval":
            stop = objfunc.counter >= self.Neval
        elif self.stop_cond == "ngen":
            stop = gen >= self.Ngen
        elif self.stop_cond == "time_limit":
            stop = time.time()-time_start >= self.time_limit
        elif self.stop_cond == "fit_target":
            if objfunc.opt == "max":
                stop = self.best_solution()[1] >= self.fit_target
            else:
                stop = self.best_solution()[1] <= self.fit_target

        return stop
    

    def get_progress(self, gen, time_start, objfunc):
        """
        Given the state of the algorithm, returns a number between 0 and 1 indicating 
        how close to the end of the algorithm we are, 0 when starting and 1 when finished.
        """

        prog = 0
        if self.stop_cond == "neval":
            prog = objfunc.counter/self.Neval
        elif self.stop_cond == "ngen":
            prog = gen/self.Ngen 
        elif self.stop_cond == "time_limit":
            prog = (time.time()-time_start)/self.time_limit
        elif self.stop_cond == "fit_target":
            best_fitness = self.best_solution()[1]
            if objfunc.opt == "max":
                prog = best_fitness/self.fit_target
            else:
                if best_fitness == 0:
                    best_fitness = 1e-40
                prog = self.fit_target/best_fitness

        return prog


    def update(self, gen, time_start, objfunc):
        """
        Given the state of the algorithm, returns a number between 0 and 1 indicating 
        how close to the end of the algorithm we are, 0 when starting and 1 when finished.
        """

        self.progress = self.get_progress(gen, time_start, objfunc)
        
        self.ended = self.stopping_condition(gen, time_start, objfunc)
    
    
    def initialize(self, objfunc):
        """
        Generates a random population of individuals
        """

        self.restart()
        self.search_strategy.initialize(objfunc)
    
    @abstractmethod
    def step(self, objfunc, time_start=0, verbose=False) -> Tuple[Indiv, float]:        
        """
        Performs a step in the algorithm
        """

    
    def optimize(self, objfunc):
        """
        Execute the algorithm to get the best solution possible along with it's evaluation
        """

        self.steps = 0

        # initialize clocks
        time_start = time.process_time()
        real_time_start = time.time()
        display_timer = time.time()

        # Initizalize search strategy 
        self.initialize(objfunc)

        # Search untill the stopping condition is met
        self.update(self.steps, real_time_start, objfunc)
        while not self.ended:

            self.step(objfunc, real_time_start)

            # Display information
            if self.verbose and time.time() - display_timer > self.v_timer:
                self.step_info(objfunc, real_time_start)
                display_timer = time.time()
        
        # Store the time spent optimizing
        self.real_time_spent = time.time() - real_time_start
        self.time_spent = time.process_time() - time_start
        
        return self.best_solution()
    
    @abstractmethod
    def step_info(self, objfunc, start_time):
        """
        Displays information about the current state of the algotithm
        """
    
    @abstractmethod
    def display_report(self, objfunc, show_plots=True):
        """
        Shows a summary of the execution of the algorithm
        """