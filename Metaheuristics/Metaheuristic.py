import numpy as np
from matplotlib import pyplot as plt
import time


class Metaheuristic:
    """
    General framework for metaheuristic algorithms
    """
    
    def __init__(self, name, objfunc, params):
        """
        Constructor of the Metaheuristic class
        """

        self.name = name
        self.params = params

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
        self.history = []
        self.best_fitness = 0
        self.time_spent = 0
        self.real_time_spent = 0


    def restart(self):
        """
        Resets the internal values of the algorithm and the number of evaluations of the fitness function.
        """

        self.history = []
        self.best_fitness = 0
        self.time_spent = 0
        self.real_time_spent = 0
        self.objfunc.counter = 0
    
    
    def save_solution(self, file_name="solution.csv"):
        """
        Save the result of an execution to a csv file in disk.
        """

        ind, fit = self.population.best_solution()
        np.savetxt(file_name, ind.reshape([1, -1]), delimiter=',')
        with open(file_name, "a") as file:
            file.write(str(fit))
    
    
    def best_solution(self):
        """
        Returns the best solution so far in the population.
        """

        return self.population.best_solution()

    
    def step(self, progress):
        """
        Does one step of the algorithm
        """
        
        _, best_fitness = self.population.best_solution()
        self.history.append(best_fitness)
    
    
    def stopping_condition(self, gen, time_start):
        """
        Given the state of the algorithm, returns wether we have finished or not.
        """

        stop = True
        if self.stop_cond == "neval":
            stop = self.objfunc.counter >= self.Neval
        elif self.stop_cond == "ngen":
            stop = gen >= self.Ngen
        elif self.stop_cond == "time_limit":
            stop = time.time()-time_start >= self.time_limit
        elif self.stop_cond == "fit_target":
            if self.objfunc.opt == "max":
                stop = self.population.best_solution()[1] >= self.fit_target
            else:
                stop = self.population.best_solution()[1] <= self.fit_target

        return stop

    
    def progress(self, gen, time_start):
        """
        Given the state of the algorithm, returns a number between 0 and 1 indicating 
        how close to the end of the algorithm we are, 0 when starting and 1 when finished.
        """

        prog = 0
        if self.stop_cond == "neval":
            prog = self.objfunc.counter/self.Neval
        elif self.stop_cond == "ngen":
            prog = gen/self.Ngen 
        elif self.stop_cond == "time_limit":
            prog = (time.time()-time_start)/self.time_limit
        elif self.stop_cond == "fit_target":
            if self.objfunc.opt == "max":
                prog = self.population.best_solution()[1]/self.fit_target
            else:
                prog = self.fit_target/self.population.best_solution()[1]

        return prog

    
    def optimize(self):
        """
        Execute the algorithm to get the best solution possible along with it's evaluation
        """

        gen = 0
        time_start = time.process_time()
        real_time_start = time.time()
        display_timer = time.time()

        self.population.generate_random()
        while not self.stopping_condition(gen, real_time_start):
            prog = self.progress(gen, real_time_start)
            self.step(prog)
            gen += 1
            if self.verbose and time.time() - display_timer > self.v_timer:
                self.step_info(gen, real_time_start)
                display_timer = time.time()
                
        self.real_time_spent = time.time() - real_time_start
        self.time_spent = time.process_time() - time_start
        return self.population.best_solution()
    
    
    def step_info(self, gen, start_time):
        """
        Displays information about the current state of the algotithm
        """

        print(f"Optimizing {self.objfunc.name}:")
        print(f"\tTime Spent {round(time.time() - start_time,2)} s")
        print(f"\tGeneration: {gen}")
        best_fitness = self.population.best_solution()[1]
        print(f"\tBest fitness: {best_fitness}")
        print(f"\tEvaluations of fitness: {self.objfunc.counter}")
        print()
    
    
    def display_report(self, show_plots=True):
        """
        Shows a summary of the execution of the algorithm
        """
        
        # Print Info
        print("Number of generations:", len(self.history))
        print("Real time spent: ", round(self.real_time_spent, 5), "s", sep="")
        print("CPU time spent: ", round(self.time_spent, 5), "s", sep="")
        print("Number of fitness evaluations:", self.objfunc.counter)
        
        best_fitness = self.population.best_solution()[1]
        print("Best fitness:", best_fitness)

        if show_plots:
            # Plot fitness history
            plt.axhline(y=0, color="black", alpha=0.9)
            plt.axvline(x=0, color="black", alpha=0.9)            
            plt.plot(self.history, "blue")
            plt.xlabel("generations")
            plt.ylabel("fitness")
            plt.title(f"{self.name} fitness")
            plt.show()