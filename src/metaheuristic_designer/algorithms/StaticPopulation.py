from __future__ import annotations
from ..ParamScheduler import ParamScheduler
from ..selectionMethods import SurvivorSelection, ParentSelection
from ..Algorithm import Algorithm
from ..Operator import Operator


class StaticPopulation(Algorithm):
    """
    Population-based algorithm where each individual is iteratively evolved with a given operator
    """

    def __init__(
        self,
        pop_init: Initializer,
        operator: Operator,
        parent_sel_op: ParentSelection = None,
        selection_op: SurvivorSelection = None,
        params: Union[ParamScheduler, dict] = {},
        name: str = "Static Population Evolution",
    ):
        self.params = params
        self.operator = operator

        if parent_sel_op is None:
            parent_sel_op = ParentSelection("Nothing")
        self.parent_sel_op = parent_sel_op

        if selection_op is None:
            selection_op = SurvivorSelection("Generational")
        self.selection_op = selection_op

        self.best = None

        super().__init__(pop_init, params=params, name=name)

    def select_parents(self, population, **kwargs):
        return self.parent_sel_op(population)

    def perturb(self, parent_list, objfunc, **kwargs):
        offspring = []
        for indiv in parent_list:
            # Apply operator
            new_indiv = self.operator(
                indiv, parent_list, objfunc, self.best, self.pop_init
            )
            new_indiv.genotype = objfunc.repair_solution(new_indiv.genotype)
            new_indiv.speed = objfunc.repair_speed(new_indiv.speed)

            # Add to offspring list
            offspring.append(new_indiv)

        # Update best solution
        current_best = max(offspring, key=lambda x: x.fitness)
        if self.best.fitness < current_best.fitness:
            self.best = current_best

        return offspring

    def select_individuals(self, population, offspring, **kwargs):
        return self.selection_op(population, offspring)

    def update_params(self, **kwargs):
        progress = kwargs["progress"]

        if isinstance(self.operator, Operator):
            self.operator.step(progress)

        self.selection_op.step(progress)

        if self.param_scheduler:
            self.param_scheduler.step(progress)
            self.params = self.param_scheduler.get_params()