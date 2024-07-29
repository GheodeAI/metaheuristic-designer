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

        if survivor_sel is None:
            survivor_sel = SurvivorSelection("KeepBest", {"amount": 1})

        super().__init__(initializer, operator=operator, survivor_sel=survivor_sel, params=params, name=name)

    def perturb(self, indiv_list, objfunc, **kwargs):
        offspring = indiv_list*self.iterations
        offspring = self.operator(offspring, objfunc, self.best, self.initializer)
        offspring = self.repair_population(offspring, objfunc)

        return offspring

    def update_params(self, **kwargs):
        super().update_params(**kwargs)

        progress = kwargs["progress"]

        if isinstance(self.operator, Operator):
            self.operator.step(progress)
