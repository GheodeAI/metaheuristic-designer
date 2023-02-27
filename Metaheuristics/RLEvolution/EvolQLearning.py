import math
import random
import numpy as np
import scipy
import scipy.special
import time

from ..Individual import *
from ...ParamScheduler import ParamScheduler

"""
Population based evolutive algorithm based on Reinforcement learning
"""
class EvolQLearning:
    def __init__(self, objfunc, operators, params, q_table= None, population=None):
        self.params = params

        self.size = params["popSize"]
        self.discount = params["discount"]
        self.alpha = params["alpha"]
        
        self.eps_range = None
        if type(params["eps"]) is list or type(params["eps"]) is tuple:
            self.eps_range = params["eps"]
            self.eps = params["eps"][0]
        else:
            self.eps = params["eps"]
        
        self.sel_exp = params["sel_exp"]

        self.objfunc = objfunc

        self.nstates = params["nstates"]
        self.actions = operators

        self.accept_rate = 0

        self.curr_action = 0
        self.curr_state = 0
        self.prev_state = 0
        self.prev_action = 0
        
        self.new_ind = None

        if q_table is None:
            self.q_table = np.zeros([self.nstates, len(self.actions)])
        else:
            self.q_table = q_table
        self.policy = np.ones(self.q_table.shape)/len(self.actions)

        self.prev_metric = 0

        if population is None:
            self.population = []
        else:
            self.population = population
    
    def step(self, progress):
        for op in self.actions:
            op.step(progress)

        if isinstance(self.params, ParamScheduler):
            self.params.step(progress)
            self.eps = params["eps"]
            self.discount = params["discount"]
            self.alpha = params["alpha"]

    """
    Gives the best solution found by the algorithm and its fitness
    """
    def best_solution(self):
        best_solution = sorted(self.population, reverse=True, key = lambda c: c.fitness)[0]
        best_fitness = best_solution.fitness
        if self.objfunc.opt == "min":
            best_fitness *= -1
        return (best_solution.vector, best_fitness)

    """
    Generates a random population of individuals
    """
    def generate_random(self):
        self.population = []
        for i in range(self.size):
            new_ind = Indiv(self.objfunc, self.objfunc.random_solution())
            self.population.append(new_ind)
    
    def update(self, progress):
        # update epsilon
        if self.eps_range is not None:
            self.eps = (1-progress) * self.eps_range[0] + progress * self.eps_range[1]
        
        # update state
        self.prev_state = self.curr_state
        self.curr_state = max(math.floor(progress * self.nstates - 1e-5), 0)

        # accept rate
        self.accept_rate = 1 - np.exp(self.sel_exp*progress)/np.exp(self.sel_exp)


    """
    Takes an action, using an operator given by the policy
    """
    def evolve(self):
        # select a new action (or operator) and store the previous one
        self.prev_action = self.curr_action
        self.curr_action = random.choices(range(len(self.actions)), weights=self.policy[self.curr_state, :])[0]
        
        # choose individual to apply the operator to
        fitness_list = [i.fitness for i in self.population]
        best_idx = np.argsort(fitness_list)[len(self.population)-(self.size//2):]
        chosen_ind = self.population[random.choice(best_idx)]

        # create the new individual applying the operator
        new_solution = self.actions[self.curr_action].evolve(chosen_ind, self.population, self.objfunc)
        new_solution = self.objfunc.check_bounds(new_solution)
        self.new_ind = Indiv(self.objfunc, new_solution)

    """
    Inserts the new element depending on the fitness of the individual
    """
    def replace(self):
        reward = 0

        # evaluate the improvement of the new solution
        fitness_list = [i.fitness for i in self.population]
        if self.prev_metric is None:
            self.prev_metric = max(fitness_list)
        
        new_fitness = self.new_ind.fitness
        reward = new_fitness - self.prev_metric
        
        # make a probabilistic (1+n) selection based on simulated annealing
        best_fit = max(fitness_list)
        if new_fitness >= best_fit or random.random() < self.accept_rate:
            self.population.pop(fitness_list.index(min(fitness_list)))
            self.population.append(self.new_ind)
        
        # update the Q table
        self.update_table(reward)
        
    
    """
    Updates the Q table
    """
    def update_table(self, reward):
        s_0 = self.prev_state
        a_0 = self.prev_action
        s_1 = self.curr_state

        # use the Q-learning update rule for the q-table
        best_q = self.q_table[s_1, :].max()
        self.q_table[s_0, a_0] += self.alpha * (reward + self.discount * best_q - self.q_table[s_0, a_0])
        
        # update the policy with the new data from the q-table
        self.policy[self.curr_state, :] = scipy.special.softmax(self.q_table[s_1, :])
        if min(self.policy[self.curr_state, :]) < self.eps:
            self.policy[self.curr_state, :] += self.eps*len(self.actions)/2
            self.policy[self.curr_state, :] /= self.policy[self.curr_state, :].sum()

    