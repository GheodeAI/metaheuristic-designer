import random
import numpy as np
from numba import jit

from ..Individual import *
from ...ParamScheduler import ParamScheduler


class CoralPopulation:    
    """
    Population of corals
    """

    def __init__(self, objfunc, substrates, params, population=None):
        """
        Constructor of the Coral Population class
        """

        self.params = params

        # Hyperparameters of the algorithm
        self.size = params["popSize"] if "popSize" in params else 100
        self.rho = params["rho"] if "rho" in params else 0.6
        self.Fb = params["Fb"] if "Fb" in params else 0.98
        self.Fd = params["Fd"] if "Fd" in params else 0.1
        self.Pd = params["Pd"] if "Pd" in params else 1
        self.k = params["k"] if "k" in params else 3
        self.K = params["K"] if "K" in params else 10
        self.group_subs = params["group_subs"] if "group_subs" in params else True

        # Dynamic parameters
        self.dynamic = params["dynamic"] if "dynamic" in params else True
        self.dyn_method = params["dyn_method"] if "dyn_method" in params else "fit"
        self.dyn_metric = params["dyn_metric"] if "dyn_metric" in params else "best"
        self.dyn_steps = params["dyn_steps"] if "dyn_steps" in params else 100
        self.prob_amp = params["prob_amp"] if "prob_amp" in params else 0.1

        # Verbose parameters
        self.verbose = params["verbose"] if "verbose" in params else True
        self.v_timer = params["v_timer"] if "v_timer" in params else 2

        # Data structures of the algorithm
        self.objfunc = objfunc
        self.substrates = substrates

        # Population initialization
        if population is None:
            self.population = []

        # Substrate data structures
        self.substrate_list = [i%len(substrates) for i in range(self.size)]
        self.substrate_weight = [1/len(substrates)]*len(substrates)

        # Dynamic data structures
        self.substrate_data = [[] for i in substrates]
        if self.dyn_method == "success":
            for idx, _ in enumerate(self.substrate_data):
                self.substrate_data[idx].append(0)
            self.larva_count = [0 for i in substrates]
        elif self.dyn_method == "diff":
            self.substrate_metric_prev = [0]*len(substrates)
        self.substrate_w_history = []
        self.subs_steps = 0
        self.substrate_metric = [0]*len(substrates)
        self.substrate_history = []

        self.prob_amp_warned = False

        # Optimization for extreme depredation
        self.updated = False
        self.identifier_list = []


    def step(self, progress):
        """
        Updates the parameters and the operators
        """

        for subs in self.substrates:
            subs.step(progress)

        if isinstance(self.params, ParamScheduler):
            self.params.step(progress)
            self.size = self.params["popSize"]
            self.rho = self.params["rho"]
            self.Fb = self.params["Fb"]
            self.Fd = self.params["Fd"]
            self.Pd = self.params["Pd"]
            self.k = self.params["k"]
            self.K = self.params["K"]

            self.prob_amp = self.params["prob_amp"]

    
    def best_solution(self):
        """
        Gives the best solution found by the algorithm and its fitness
        """

        best_solution = sorted(self.population, reverse=True, key = lambda c: c.fitness)[0]
        best_fitness = best_solution.fitness
        if self.objfunc.opt == "min":
            best_fitness *= -1
        return (best_solution.vector, best_fitness)

    
    def generate_random(self):
        """
        Generates a random population of corals
        """

        amount = int(self.size*self.rho) - len(self.population)

        for i in range(amount):
            substrate_idx = self.substrate_list[i]
            new_sol = self.objfunc.random_solution()
            fixed_sol = self.objfunc.check_bounds(new_sol)
            new_coral = Indiv(self.objfunc, fixed_sol, operator=self.substrates[substrate_idx])
            self.population.append(new_coral)

    
    # def insert_solution(self, solution, mutate=False, strength=0.1):
    #     #solution = self.objfunc.check_bounds(solution)
    #     if mutate:
    #         solution = gaussian(solution, strength)
    #         solution = self.objfunc.check_bounds(solution)
        
    #     if len(self.population) < self.size:
    #         new_ind = Coral(solution, self.objfunc)
    #         self.population.append(new_ind)
    #     else:
    #         new_ind = Coral(solution, self.objfunc)
    #         idx = random.randint(self.size)
    #         self.population[idx] = new_ind
    
    def get_value_from_data(self, data):
        """
        Evaluates the data obtained by an operator
        """

        result = 0

        # Choose what information to extract from the data gathered
        if len(data) > 0:
            data = sorted(data)
            if self.dyn_metric == "best":
                result = max(data)
            elif self.dyn_metric == "avg":
                result = sum(data)/len(data)
            elif self.dyn_metric == "med":
                if len(data) % 2 == 0:
                    result = (data[len(data)//2-1]+data[len(data)//2])/2
                else:
                    result = data[len(data)//2]
            elif self.dyn_metric == "worse":
                result = min(data)
        
        return result

    
    def evaluate_substrates(self):
        """
        Evaluates the substrates using a given metric
        """

        metric = 0
        
        # take reference data for the calculation of the difference of the next evaluation
        if self.dyn_method == "diff":
            full_data = [d for subs_data in self.substrate_data for d in subs_data]
            metric = self.get_value_from_data(full_data)
        
        # calculate the value of each substrate with the data gathered
        for idx, s_data in enumerate(self.substrate_data):
            if self.dyn_method == "success":

                # obtain the rate of success of the larvae
                if self.larva_count[idx] > 0:
                    self.substrate_metric[idx] = s_data[0]/self.larva_count[idx]
                else:
                    self.substrate_metric[idx] = 0

                # Reset data for nex iteration
                self.substrate_data[idx] = [0]
                self.larva_count[idx] = 0

            elif self.dyn_method == "fitness" or self.dyn_method == "diff":

                # obtain the value used in the evaluation of the substrate 
                self.substrate_metric[idx] = self.get_value_from_data(s_data)

                # Calculate the difference of the fitness in this generation to the previous one and
                # store the current value for the next evaluation
                if self.dyn_method == "diff":
                    self.substrate_metric[idx] =  self.substrate_metric[idx] - self.substrate_metric_prev[idx]
                    self.substrate_metric_prev[idx] = metric
                
                # Reset data for next iteration
                self.substrate_data[idx] = []


    def evolve_with_substrates(self):
        """
        Evolves the population using the corresponding substrates
        """

        larvae = []

        # evolve within the substrate or mix with the whole population
        if self.group_subs:

            # Divide the population based on their substrate type
            substrate_groups = [[] for i in self.substrates]
            for i, idx in enumerate(self.substrate_list):
                if i < len(self.population):
                    substrate_groups[idx].append(self.population[i])
            
            
            # Reproduce the corals of each group
            for i, coral_group in enumerate(substrate_groups):

                # Restart fitness record if there are corals in this substrate
                for coral in coral_group:

                    # Generate new coral
                    if random.random() <= self.Fb:
                        new_coral = coral.reproduce(coral_group)
                        
                        # Get data of the current substrate
                        if self.dyn_method == "fitness" or self.dyn_method == "diff":
                            self.substrate_data[i].append(new_coral.fitness)
                    else:
                        new_sol = self.objfunc.random_solution()
                        fixed_sol = self.objfunc.check_bounds(new_sol)
                        new_coral = Indiv(self.objfunc, fixed_sol)

                    # Add larva to the list of larvae
                    larvae.append(new_coral)
        else:
            for idx, coral in enumerate(self.population):

                # Generate new coral
                if random.random() <= self.Fb:
                    new_coral = coral.reproduce(self.population)
                    
                    # Get the index of the substrate this individual belongs to
                    s_names = [i.name for i in self.substrates]
                    s_idx = s_names.index(coral.operator.name)

                    # Get data of the current substrate
                    if self.dyn_method == "fitness" or self.dyn_method == "diff":
                        self.substrate_data[s_idx].append(new_coral.fitness)
                else:
                    new_sol = self.objfunc.random_solution()
                    fixed_sol = self.objfunc.check_bounds(new_sol)
                    new_coral = Indiv(self.objfunc, fixed_sol)

                # Add larva to the list of larvae
                larvae.append(new_coral)
        
        return larvae

    
    def substrate_probability(self, values):
        """
        Converts the evaluation values of the substrates to a probability distribution
        """

        # Normalization to avoid passing big values to softmax 
        weight = np.array(values)
        if np.abs(weight).sum() != 0:
            weight = weight/np.abs(weight).sum()
        else:
            weight = weight/(np.abs(weight).sum()+1e-5)
        
        # softmax to convert to a probability distribution
        exp_vec = np.exp(weight)
        amplified_vec = exp_vec**(1/self.prob_amp)
        
        # if there are numerical error default repeat with a default value
        if (amplified_vec == 0).any() or not np.isfinite(amplified_vec).all():
            if not self.prob_amp_warned:
                print("Warning: the probability amplification parameter is too small, defaulting to prob_amp = 1")
                self.prob_amp_warned = True
            prob = exp_vec/exp_vec.sum()
        else:
            prob = amplified_vec/amplified_vec.sum()

        # If probabilities get too low, equalize them
        if (prob <= 0.02/len(values)).any():
            prob += 0.02/len(values)
            prob = prob/prob.sum()

        return prob


    def generate_substrates(self, progress=0):
        """
        Generates the assignment of the substrates
        """

        n_substrates = len(self.substrates)

        if progress > self.subs_steps/self.dyn_steps:
            self.subs_steps += 1
            self.evaluate_substrates()

        # Assign the probability of each substrate
        if self.dynamic:
            self.substrate_weight = self.substrate_probability(self.substrate_metric)
            self.substrate_w_history.append(self.substrate_weight)
        
        # Choose each substrate with the weights chosen
        self.substrate_list = random.choices(range(n_substrates), 
                                            weights=self.substrate_weight, k=self.size)

        # Assign the substrate to each coral
        for idx, coral in enumerate(self.population):
            substrate_idx = self.substrate_list[idx]
            coral.operator = self.substrates[substrate_idx]

        # save the evaluation of each substrate
        self.substrate_history.append(np.array(self.substrate_metric))


    def larvae_setting(self, larvae_list):
        """
        Inserts solutions into our reef with some conditions
        """

        s_names = [i.name for i in self.substrates]

        for larva in larvae_list:
            attempts_left = self.k
            setted = False
            idx = -1

            # Try to settle 
            while attempts_left > 0 and not setted:
                # Choose a random position
                idx = random.randrange(0, self.size)

                # If it's empty settle in there, otherwise, try
                # to replace the coral in that position
                if setted := (idx >= len(self.population)):
                    self.population.append(larva)
                elif setted := (larva.fitness > self.population[idx].fitness):
                    self.population[idx] = larva

                attempts_left -= 1
            
            if larva.operator is not None:
                s_idx = s_names.index(larva.operator.name)
                if self.dyn_method == "success":
                    self.larva_count[s_idx] += 1

            # Assign substrate to the setted coral
            if setted:
                self.updated = True
                
                # Get substrate index
                if self.dyn_method == "success" and larva.substrate is not None:
                    self.substrate_data[s_idx][0] += 1

                substrate_idx = self.substrate_list[idx]
                larva.operator = self.substrates[substrate_idx]
    
    
    def local_search(self, operator, n_ind, iterations=100):
        """
        Performs a local search with the best "n_ind" corals
        """

        fitness_values = np.array([coral.fitness for coral in self.population])
        affected_corals = list(np.argsort(fitness_values))[len(self.population)-n_ind:]
        
        for i in affected_corals:
            best = self.population[i]

            for j in range(iterations):
                new_solution = operator.evolve(self.population[i], [], self.objfunc)
                new_solution = self.objfunc.check_bounds(new_solution)
                new_coral = Indiv(self.objfunc, new_solution, operator=self.population[i].substrate)
                if new_coral.fitness > best.fitness:
                    best = new_coral
            
            self.population[i] = best

    
    def depredation(self):
        """
        Removes a portion of the worst solutions in our population
        """

        if self.Pd == 1:
            self.full_depredation()
        else:
            # Calculate the number of affected corals
            amount = int(len(self.population)*self.Fd)

            # Select the worse individuals in the grid
            fitness_values = np.array([coral.fitness for coral in self.population])
            affected_corals = list(np.argsort(fitness_values))[:amount]

            # Set a 'dead' flag in the affected corals with a small probability
            alive_count = len(self.population)

            for i in affected_corals:

                # Ensure there are at least 2 individuals in the population
                if alive_count <= 2:
                    break
                
                # Kill the indiviual with probability Pd
                dies = random.random() <= self.Pd
                self.population[i].is_dead = dies
                if dies:
                    alive_count -= 1

            # Remove the dead corals from the population
            self.population = list(filter(lambda c: not c.is_dead, self.population))

    def full_depredation(self):
        """
        Depredation when the probability of death is 1
        """
        # Calculate the number of affected corals
        amount = int(len(self.population)*self.Fd)

        # Select the worse individuals in the grid
        fitness_values = np.array([coral.fitness for coral in self.population])
        affected_corals = list(np.argsort(fitness_values))[:amount]

        # Remove all the individuals chosen
        self.population = [self.population[i] for i in range(len(self.population)) if i not in affected_corals] 

    
    def update_identifier_list(self):
        """
        Makes sure that we calculate the list of solution vectors only once
        """

        if not self.updated:
            self.identifier_list = [i.vector for i in self.population]
            self.updated = True

    
    def extreme_depredation(self, tol=0):
        """
        Eliminates duplicate solutions from our population
        """
        
        # Get a list of the vectors of each individual in the population
        self.update_identifier_list()

        # Store the individuals with vectors repeated more than K times
        repeated_idx = []
        for idx, val in enumerate(self.identifier_list):
            if np.count_nonzero((np.isclose(val,x,tol)).all() for x in self.identifier_list[:idx]) > self.K:
                repeated_idx.append(idx)
        
        # Remove the individuals selected in the previous step
        self.population = [val for idx, val in enumerate(self.population) if idx not in repeated_idx]