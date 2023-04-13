from __future__ import annotations
import numpy as np
import random
from typing import Union, List
from copy import copy
from ..Individual import Individual
from ..Operators import OperatorReal, OperatorMeta
from ..SurvivorSelection import SurvivorSelection
from ..ParentSelection import ParentSelection
from .StaticPopulation import StaticPopulation
from ..ParamScheduler import ParamScheduler
from ..Algorithm import Algorithm


class CRO_SL(Algorithm):
    def __init__(self, operator_list: List[Operator], params: Union[ParamScheduler, dict] = {}, name: str = "CRO_SL"):
        super().__init__(name)

        # Hyperparameters of the algorithm
        self.params = params
        self.maxpopsize = params["popSize"]
        self.popsize = round(params["popSize"] * params["rho"])
        self.operator_list = operator_list
        self.operator_idx = [i%len(operator_list) for i in range(params["popSize"])]

        self.selection_op = SurvivorSelection("CRO", {"Fd": params["Fd"], "Pd": params["Pd"], "attempts": params["attempts"], "maxPopSize": params["popSize"]})

        self.best = None
    
    def perturb(self, parent_list, objfunc, progress, history):
        offspring = []
        for idx, indiv in enumerate(parent_list):
            
            # Select operator
            op_idx = self.operator_idx[idx]
            op = self.operator_list[op_idx]

            # Apply operator
            new_indiv = op(indiv, parent_list, objfunc, self.best)
            new_indiv.genotype = objfunc.repair_solution(new_indiv.genotype)
            new_indiv.speed = objfunc.repair_speed(new_indiv.speed)

            # Store best vector for individual
            new_indiv.store_best(indiv)

            # Add to offspring list
            offspring.append(new_indiv)

        # Update best solution
        current_best = max(offspring, key=lambda x: x.fitness)
        if self.best.fitness < current_best.fitness:
            self.best = current_best

        return offspring
    
    def select_individuals(self, population, offspring, progress=0, history=None):
        return self.selection_op(population, offspring)

    def update_params(self, progress):
        """
        Updates the parameters and the operators
        """

        self.popsize = len(self.population)

        for op in self.operator_list:
            op.step(progress)

        if isinstance(self.params, ParamScheduler):
            self.params.step(progress)
            self.size = self.params["popSize"]
            self.n_offspring = self.params["offspringSize"]


class PCRO_SL(CRO_SL):
    def __init__(self, operator_list: List[Operator], params: Union[ParamScheduler, dict] = {}, name: str = "PCRO_SL"):
        super().__init__(operator_list, params, name)
        self.operator_idx = random.choices(range(len(self.operator_list)), k=self.popsize)

    def update_params(self, progress):
        """
        Updates the parameters and the operators
        """

        self.operator_idx = random.choices(range(len(self.operator_list)), k=self.maxpopsize)

        super().update_params(progress)