from __future__ import annotations
from typing import Union
from copy import copy
from ..ParamScheduler import ParamScheduler
from ..Algorithm import Algorithm
from ..Operator import Operator
from ..Operators import OperatorReal
from ..SelectionMethods import SurvivorSelection


class LocalSearch(Algorithm):
    """
    Local search algorithm
    """

    def __init__(self, pop_init: Initializer, perturb_op: Operator = None, selection_op: SurvivorSelection = None,
                 params: Union[ParamScheduler, dict] = {}, name: str = "LocalSearch"):
        
        self.iterations = params["iters"] if "iters" in params else 100

        if perturb_op is None:
            perturb_op = OperatorReal("Nothing")
        self.perturb_op = perturb_op

        if selection_op is None:
            selection_op = SurvivorSelection("One-to-One")
        self.selection_op = selection_op

        super().__init__(pop_init, params=params, name=name)

    def perturb(self, indiv_list, objfunc, progress=0, history=None):
        next_indiv_list = copy(indiv_list)
        for i in range(self.iterations):
            
            offspring = []
            for indiv in indiv_list:

                # Perturb individual
                new_indiv = self.perturb_op(indiv, indiv_list, objfunc, self.best, self.pop_init)
                new_indiv.genotype = objfunc.repair_solution(new_indiv.genotype)

                offspring.append(new_indiv)

            # Keep best individual regardless of selection method
            current_best = max(offspring, key=lambda x: x.fitness)
            if self.best.fitness < current_best.fitness:
                self.best = current_best

            next_indiv_list = self.selection_op(next_indiv_list, offspring)
        
        return next_indiv_list

    def update_params(self, progress=0):
        if isinstance(self.perturb_op, Operator):
            self.perturb_op.step(progress)
