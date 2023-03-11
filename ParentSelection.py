import random
import numpy as np
from typing import Union
from .ParamScheduler import *


class ParentSelection:
    """
    Operator class that has continuous mutation and cross methods
    """

    def __init__(self, name: str, params: Union[ParamScheduler, dict]=None):
        """
        Constructor for the ParentSelection class
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
    

    def __call__(self, population):
        """
        Shorthand for calling the 'select' method
        """
        
        return self.select(population)


    def step(self, progress):
        """
        Updates the parameters of the method using a paramater scheduler if it exists
        """

        if self.param_scheduler:
            self.param_scheduler.step(progress)
            self.params = self.param_scheduler.get_params()


    def select(self, population): 
        """
        Evolves a solution with a different strategy depending on the type of operator
        """
        
        result = []
        if self.name == "tournament":
            result = tournament(population, self.params["amount"], self.params["p"])
        elif self.name == "best":
            result = select_best(population, self.params["amount"])
        elif self.name == "nothing":
            result = population
        else:
            print(f"Error: parent selection method \"{self.name}\" not defined")
            exit(1)
        
        return result


def select_best(population, amount):
    amount = round(amount)
    return sorted(population, reverse=True, key = lambda x: x.fitness)[:amount]


def tournament(population, tourn_size, prob):
    parent_pool = []
    for _ in range(len(population)):
        parents = random.sample(population, tourn_size)
        fits = [i.fitness for i in parents]

        parent = None
        if random.random() < prob:
            parent = random.choice(parents)
        else:
            parent = parents[fits.index(max(fits))]
        
        parent_pool.append(parent)
    
    return parent_pool