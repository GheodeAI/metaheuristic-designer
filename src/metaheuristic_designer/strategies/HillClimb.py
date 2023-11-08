from __future__ import annotations
from typing import Union
from copy import copy
from ..ParamScheduler import ParamScheduler
from ..SearchStrategy import SearchStrategy
from ..Operator import Operator
from ..operators import OperatorNull
from ..selectionMethods import SurvivorSelection


class HillClimb(SearchStrategy):
    """
    Hill Climbing algorithm
    """

    def __init__(
        self,
        pop_init: Initializer,
        perturb_op: Operator = None,
        selection_op: SurvivorSelection = None,
        params: Union[ParamScheduler, dict] = {},
        name: str = "HillClimb",
    ):
        self.iterations = params.get("iters", 1)

        if perturb_op is None:
            perturb_op = OperatorNull()
        self.perturb_op = perturb_op

        if selection_op is None:
            selection_op = SurvivorSelection("HillClimb")
        self.selection_op = selection_op

        super().__init__(pop_init, params=params, name=name)

    def perturb(self, indiv_list, objfunc, **kwargs):
        offspring = []
        for indiv in indiv_list:
            # Perturb individual
            new_indiv = self.perturb_op(indiv, indiv_list, objfunc, self.best, self.pop_init)
            new_indiv.genotype = objfunc.repair_solution(new_indiv.genotype)
            new_indiv.speed = objfunc.repair_speed(new_indiv.speed)

            offspring.append(new_indiv)

        return offspring

    def select_individuals(self, population, offspring, **kwargs):
        return self.selection_op(population, offspring)

    def update_params(self, **kwargs):
        progress = kwargs["progress"]

        if isinstance(self.perturb_op, Operator):
            self.perturb_op.step(progress)
