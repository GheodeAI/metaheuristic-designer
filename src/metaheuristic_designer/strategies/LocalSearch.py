from __future__ import annotations
from typing import Union
from copy import copy
from ..ParamScheduler import ParamScheduler
from ..SearchStrategy import SearchStrategy
from ..Operator import Operator
from ..operators import OperatorNull
from ..selectionMethods import SurvivorSelection


class LocalSearch(SearchStrategy):
    """
    Local search algorithm
    """

    def __init__(
        self,
        initializer: Initializer,
        operator: Operator = None,
        survivor_sel: SurvivorSelection = None,
        params: ParamScheduler | dict = {},
        name: str = "LocalSearch",
    ):
        self.iterations = params.get("iters", 100)

        if operator is None:
            operator = OperatorNull()
        self.operator = operator

        if survivor_sel is None:
            survivor_sel = SurvivorSelection("KeepBest", {"amount": 1})

        super().__init__(initializer, survivor_sel=survivor_sel, params=params, name=name)

    def perturb(self, indiv_list, objfunc, **kwargs):
        offspring = []
        indiv = indiv_list[0]
        for i in range(self.iterations):
            # Perturb individual
            new_indiv = self.operator(indiv, indiv_list, objfunc, self.best, self.initializer)
            new_indiv.genotype = objfunc.repair_solution(new_indiv.genotype)
            new_indiv.speed = objfunc.repair_speed(new_indiv.speed)

            offspring.append(new_indiv)

        return offspring

    def update_params(self, **kwargs):
        super().update_params(**kwargs)

        progress = kwargs["progress"]

        if isinstance(self.operator, Operator):
            self.operator.step(progress)
