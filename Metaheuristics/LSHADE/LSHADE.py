import numpy as np
from matplotlib import pyplot as plt
import pyade
import pyade.lshade
import pyade.ilshade
import time


class LSHADE:
    """
    LSHADE optimization algorithm
    """

    def __init__(self, objfunc, params):
        """
        Constructor of the LSHADE algorithm
        """

        self.name = "LSHADE"
        self.params = params
        self.objfunc = objfunc

        self.best_solution = None

        self.real_time_spent = 0
        self.time_spent = 0

        # Verbose parameters
        self.verbose = params["verbose"] if "verbose" in params else True
        self.v_timer = params["v_timer"] if "v_timer" in params else 1

        # Stopping conditions
        self.stop_cond = params["stop_cond"] if "stop_cond" in params else "time_limit"
        self.ngen = params["ngen"] if "ngen" in params else 100
        self.neval = params["neval"] if "neval" in params else 1e5
        self.time_limit = params["time_limit"] if "time_limit" in params else 10.0
        self.fit_target = params["fit_target"] if "fit_target" in params else 0
    

    def restart(self):
        """
        Resets the data of the algorithm
        """

        pass # Not implemented


    def step(self, progress, depredate=True, classic=False):
        """
        One step of the algorithm
        """

        pass # Not implemented
    
    
    def stopping_condition(self, gen, time_start):
        """
        Stopping conditions given by a parameter
        """

        stop = True
        if self.stop_cond == "neval":
            stop = self.objfunc.counter >= self.Neval
        elif self.stop_cond == "ngen":
            stop = gen >= self.Ngen
        elif self.stop_cond == "time":
            stop = time.time()-time_start >= self.time_limit
        elif self.stop_cond == "fit_target":
            if self.objfunc.opt == "max":
                stop = self.population.best_solution()[1] >= self.fit_target
            else:
                stop = self.population.best_solution()[1] <= self.fit_target

        return stop

    
    def progress(self, gen, time_start):
        """
        Progress of the algorithm as a number between 0 and 1, 0 at the begining, 1 at the end
        """

        pass # Not implemented

    
    def optimize(self):
        """
        Execute the algorithm to get the best solution possible along with it's evaluation
        """

        # You may want to use a variable so its easier to change it if we want
        algorithm = pyade.lshade 

        # We get default parameters for a problem with two variables
        lshade_params = algorithm.get_default_params(dim=int(self.objfunc.size))
        lshade_params["max_evals"] = int(self.neval)

        # We define the boundaries of the variables
        lshade_params['bounds'] = np.array([[-10, 10]] * self.objfunc.size) 

        # We indicate the function we want to minimize
        lshade_params['func'] = lambda x: -self.objfunc.fitness(x)

        time_start = time.process_time()
        real_time_start = time.time()

        # We run the algorithm and obtain the results
        self.best_solution, self.best_fitness = algorithm.apply(**lshade_params)

        self.time_spent = time.process_time() - time_start
        self.real_time_spent = time.time() - real_time_start
        
        return self.best_solution, self.best_fitness
    
    
    def step_info(self, gen, start_time):
        """
        Displays information about the current state of the algotithm
        """

        pass # Not implemented
    
    
    def display_report(self, show_plots=True):
        """
        Shows a summary of the execution of the algorithm
        """
        
        print("Real time spent: ", round(self.real_time_spent, 5), "s", sep="")
        print("CPU time spent: ", round(self.time_spent, 5), "s", sep="")
        print("Number of fitness evaluations:", self.objfunc.counter)
        print("Best fitness:", self.best_fitness)