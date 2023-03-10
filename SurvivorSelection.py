import random
import numpy as np

from .ParamScheduler import *


class SurvivorSelection:
    """
    Operator class that has continuous mutation and cross methods
    """

    def __init__(self, name, params = {}):
        """
        Constructor for the SurvivorSelection class
        """

        self.name = name.lower()

        self.param_scheduler = None
        if params is None:
            self.params = {"amount": 10, "p":0.1}
        elif isinstance(params, ParamScheduler):
            self.param_scheduler = params
            self.params = self.param_scheduler.get_params()
        else:
            self.params = params
    

    def __call__(self, popul, offspring):
        """
        Shorthand for calling the 'select' method
        """

        return self.select(popul, offspring)
    

    def step(self, progress):
        """
        Updates the parameters of the method using a paramater scheduler if it exists
        """

        if self.param_scheduler:
            self.param_scheduler.step(progress)
            self.params = self.param_scheduler.get_params()
    
    
    def select(self, popul, offspring):     
        """
        Takes a population with its offspring and returns the individuals that survive
        to produce the next generation.
        """   

        result = []
        if self.name == "elitism":
            result = elitism(popul, offspring, self.params["amount"])
        elif self.name == "condElitism":
            result = cond_elitism(popul, offspring, self.params["amount"])
        elif self.name == "Generational":
            result = offspring
        elif self.name == "One-to-one":
            result = one_to_one(popul, offspring)
        elif self.name == "(m+n)":
            result = lamb_plus_mu(popul, offspring)
        elif self.name == "(m,n)":
            result = lamb_comma_mu(popul, offspring)
            # divide in parts
        else:
            print(f"Error: parent selection method \"{self.name}\" not defined")
            exit(1)
        
        return result


def one_to_one(popul, offspring):
    new_population = []
    for parent, child in zip(popul, offspring):
        if child.fitness > parent.fitness:
            new_population.append(child)
        else:
            new_population.append(parent)
    return new_population


def elitism(popul, offspring, amount):
    amount = round(amount)
    best_parents = sorted(popul, reverse=True, key = lambda x: x.fitness)[:amount]
    new_offspring = sorted(offspring, reverse=True, key = lambda x: x.fitness)[:len(popul)-amount] + best_parents
    return new_offspring


def cond_elitism(popul, offspring, amount):
    best_parents = sorted(popul, reverse=True, key = lambda x: x.fitness)[:amount]
    new_offspring = sorted(offspring, reverse=True, key = lambda x: x.fitness)[:len(popul)]
    best_offspring = new_offspring[:amount]

    for idx, val in enumerate(best_parents):
        if val.fitness > best_offspring[0].fitness:
            new_offspring.pop()
            new_offspring = [val] + new_offspring
    
    return new_offspring


def lamb_plus_mu(popul, offspring):
    population = popul + offspring
    return sorted(population, reverse=True, key = lambda x: x.fitness)[:len(popul)]


def lamb_comma_mu(popul, offspring):
    return sorted(offspring, reverse=True, key = lambda x: x.fitness)[:len(popul)]