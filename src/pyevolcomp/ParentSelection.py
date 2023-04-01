from __future__ import annotations
from typing import Union, List
from .ParamScheduler import *
import random
import numpy as np


_parent_sel_methods = [
    "tournament",
    "best",
    "nothing"   
]


class ParentSelection:
    """
    Operator class that has continuous mutation and cross methods
    """

    def __init__(self, name: str, params: Union[ParamScheduler, dict]=None):
        """
        Constructor for the ParentSelection class
        """

        self.name = name.lower()

        if name.lower() not in _parent_sel_methods:
            raise ValueError(f"Parent selection method \"{self.name}\" not defined")
        
        self.param_scheduler = None
        if params is None:
            self.params = {"amount": 10, "p":0.1}
        elif isinstance(params, ParamScheduler):
            self.param_scheduler = params
            self.params = self.param_scheduler.get_params()
        else:
            self.params = params
    

    def __call__(self, population: List[Individual]):
        """
        Shorthand for calling the 'select' method
        """
        
        return self.select(population)


    def step(self, progress: float):
        """
        Updates the parameters of the method using a paramater scheduler if it exists
        """

        if self.param_scheduler:
            self.param_scheduler.step(progress)
            self.params = self.param_scheduler.get_params()

            if "amount" in self.params:
                self.params["amount"] = round(self.params["amount"])


    def select(self, population: List[Individual]) -> List[Individual]: 
        """
        Selects a subsection of the population along with the indices of each individual in the original population
        """
        
        parents = []
        order = []
        if self.name == "tournament":
            parents, order = prob_tournament(population, self.params["amount"], self.params["p"])

        elif self.name == "best":
            parents, order = select_best(population, self.params["amount"])

        elif self.name == "nothing":
            parents, order = population, range(len(population))
        
        return parents, order


def select_best(population, amount):
    """
    Selects the best parent of the population as parents
    """

    # Get the fitness of all the individuals
    fitness_list = np.fromiter(map(lambda x: x.fitness, population), dtype=float)

    # Get the index of the individuals sorted by fitness 
    order = np.argsort(fitness_list)[::-1][:amount]
    
    # Select the 'amount' best individuals
    parents = [population[i] for i in order]

    return parents, order


def prob_tournament(population, tourn_size, prob):
    """
    Selects the parents for the next generation by tournament
    """

    parent_pool = []
    order = []
    

    for _ in population:

        # Choose 'tourn_size' individuals for the torunament
        parent_idxs = random.sample(range(len(population)), tourn_size)
        parents = [population[i] for i in parent_idxs]
        fits = [i.fitness for i in parents]

        # Choose one of the individuals
        if random.random() < prob:
            idx = random.randint(0,tourn_size-1)
        else:
            idx = fits.index(max(fits))

        # Add the individuals to the list
        order.append(parent_idxs[idx])
        parent = parents[idx]
        parent_pool.append(parent)

    return parent_pool, order