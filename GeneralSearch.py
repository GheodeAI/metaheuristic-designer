import numpy as np
from matplotlib import pyplot as plt
from typing import Union
from .ParamScheduler import ParamScheduler
from .Algorithms import BaseAlgorithm
import time


class GeneralSearch:
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
    

    def step(self, objfunc, time_start=0, verbose=False):
        """
        Performs a step in the algorithm
        """

        # Do a search step
        population = self.search_strategy.population
        


        parents, parent_idxs = self.search_strategy.select_parents(population, self.progress, self.best_history)

        offspring = self.search_strategy.perturb(parents, objfunc, self.progress, self.best_history)

        population = self.search_strategy.select_individuals(population, offspring, self.progress, self.best_history)

        self.search_strategy.population = population
        
        best_individual, best_fitness = self.search_strategy.best_solution()
        self.search_strategy.update_params(self.progress)
        self.steps += 1
            
        # Store information
        self.best_history.append(best_individual)
        self.fit_history.append(best_fitness)

        # Display information
        if verbose:
            self.step_info(time_start)
        
        # Update internal state
        self.update(self.steps, time_start, objfunc)
        
        return (best_individual, best_fitness)

    
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
        self.search_strategy.initialize(objfunc)

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
    
    
    def step_info(self, objfunc, start_time):
        """
        Displays information about the current state of the algotithm
        """

        print(f"Optimizing {objfunc.name} using {self.search_strategy.name}:")
        print(f"\tTime Spent {round(time.time() - start_time,2)} s")
        print(f"\tGeneration: {self.steps}")
        best_fitness = self.best_solution()[1]
        print(f"\tBest fitness: {best_fitness}")
        print(f"\tEvaluations of fitness: {objfunc.counter}")
        self.search_strategy.extra_step_info()
        print()
    
    
    def display_report(self, objfunc, show_plots=True):
        """
        Shows a summary of the execution of the algorithm
        """
        
        # Print Info
        print("Number of generations:", len(self.fit_history))
        print("Real time spent: ", round(self.real_time_spent, 5), "s", sep="")
        print("CPU time spent: ", round(self.time_spent, 5), "s", sep="")
        print("Number of fitness evaluations:", objfunc.counter)
        
        best_fitness = self.best_solution()[1]
        print("Best fitness:", best_fitness)

        if show_plots:
            # Plot fitness history
            plt.axhline(y=0, color="black", alpha=0.9)
            plt.axvline(x=0, color="black", alpha=0.9)            
            plt.plot(self.fit_history, "blue")
            plt.xlabel("generations")
            plt.ylabel("fitness")
            plt.title(f"{self.search_strategy.name} fitness")
            plt.show()
        
        self.search_strategy.extra_report(show_plots)