from __future__ import annotations
from typing import Union, List, Tuple
import random
import numpy as np
from typing import Union
from .ParamScheduler import *

_surv_methods = [
    "elitism",
    "condelitism",
    "generational",
    "one-to-one",
    "(m+n)",
    "(m,n)",
    "nothing"
]

class SurvivorSelection:
    """
    Operator class that has continuous mutation and cross methods
    """

    def __init__(self, name: str, params: Union[ParamScheduler, dict]=None, padding=False):
        """
        Constructor for the SurvivorSelection class
        """

        self.name = name.lower()

        if name.lower() not in _surv_methods:
            raise ValueError(f"Survivor selection method \"{self.name}\" not defined")

        self.param_scheduler = None
        if params is None:
            self.params = {"amount": 10, "p":0.1}
        elif isinstance(params, ParamScheduler):
            self.param_scheduler = params
            self.params = self.param_scheduler.get_params()
        else:
            self.params = params
    

    def __call__(self, popul: List[Individual], offspring: List[Individual]) -> List[Individual]:
        """
        Shorthand for calling the 'select' method
        """

        return self.select(popul, offspring)
    

    def step(self, progress: float):
        """
        Updates the parameters of the method using a paramater scheduler if it exists
        """

        if self.param_scheduler:
            self.param_scheduler.step(progress)
            self.params = self.param_scheduler.get_params()

            if "amount" in self.params:
                self.params["amount"] = round(self.params["amount"])

    

    def get_state(self):
        """
        Gets the current state of the algorithm as a dictionary.
        """

        data = {
            "name": self.name
        }

        if self.param_scheduler:
            data["param_scheduler"] = self.param_scheduler.get_state()
            data["params"] = self.param_scheduler.get_params()
        else:
            data["params"] = self.params
        
        return data
    
    
    def select(self, popul: List[Individual], offspring: List[Individual]) -> List[Individual]:     
        """
        Takes a population with its offspring and returns the individuals that survive
        to produce the next generation.
        """   

        result = []
        if self.name == "elitism":
            result = elitism(popul, offspring, self.params["amount"])

        elif self.name == "condelitism":
            result = cond_elitism(popul, offspring, self.params["amount"])

        elif self.name == "generational" or self.name == "nothing":
            result = offspring

        elif self.name == "one-to-one":
            result = one_to_one(popul, offspring)

        elif self.name == "(m+n)":
            result = lamb_plus_mu(popul, offspring)

        elif self.name == "(m,n)":
            result = lamb_comma_mu(popul, offspring)
        
        return result


def one_to_one(popul, offspring):
    new_population = []
    for parent, child in zip(popul, offspring):
        if child.fitness > parent.fitness:
            new_population.append(child)
        else:
            new_population.append(parent)
    
    if len(offspring) < len(popul):
        n_leftover = len(offspring) - len(popul)
        new_population += popul[n_leftover:]

    return new_population


def elitism(popul, offspring, amount):
    selected_offspring = sorted(offspring, reverse=True, key = lambda x: x.fitness)[:len(popul)-amount]
    best_parents = sorted(popul, reverse=True, key = lambda x: x.fitness)[:amount]

    return best_parents + selected_offspring


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