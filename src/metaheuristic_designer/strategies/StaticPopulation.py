from __future__ import annotations
from ..ParamScheduler import ParamScheduler
from ..selectionMethods import SurvivorSelection, ParentSelection, SurvivorSelectionNull, ParentSelectionNull
from ..SearchStrategy import SearchStrategy
from ..Operator import Operator


class StaticPopulation(SearchStrategy):
    """
    Population-based algorithm where each individual is iteratively evolved with a given operator
    """

    def __init__(
        self,
        initializer: Initializer,
        operator: Operator,
        parent_sel: ParentSelection = None,
        survivor_sel: SurvivorSelection = None,
        params: ParamScheduler | dict = {},
        name: str = "Static Population Evolution",
    ):
        self.operator = operator

        super().__init__(initializer, parent_sel=parent_sel, survivor_sel=survivor_sel, params=params, name=name)

    def perturb(self, parent_list, objfunc, **kwargs):
        offspring = []
        for indiv in parent_list:
            # Apply operator
            new_indiv = self.operator(indiv, parent_list, objfunc, self.best, self.initializer)
            new_indiv.genotype = objfunc.repair_solution(new_indiv.genotype)
            new_indiv.speed = objfunc.repair_speed(new_indiv.speed)

            # Add to offspring list
            offspring.append(new_indiv)

        return offspring

    def update_params(self, **kwargs):
        super().update_params(**kwargs)

        progress = kwargs["progress"]

        if isinstance(self.operator, Operator):
            self.operator.step(progress)

        if isinstance(self.survivor_sel, SurvivorSelection):
            self.survivor_sel.step(progress)

        if self.param_scheduler:
            self.param_scheduler.step(progress)
            self.params = self.param_scheduler.get_params()
