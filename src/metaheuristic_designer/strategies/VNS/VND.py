from __future__ import annotations
from typing import Union
from copy import copy
import warnings
from ...ParamScheduler import ParamScheduler
from ...SearchStrategy import SearchStrategy
from ...Operator import Operator
from ...operators import OperatorMeta
from ...selectionMethods import SurvivorSelection
from .vns_neighborhood_changes import *


class VND(SearchStrategy):
    """
    Variable Neighborhood Descent

    As seen in:
        Hansen, P., & Mladenovic, N. (2003). A tutorial on variable neighborhood search. Les Cahiers du GERAD ISSN, 711, 2440.
    """

    def __init__(
        self,
        pop_init: Initializer,
        op_list: List[Operator],
        selection_op: SurvivorSelection = None,
        params: Union[ParamScheduler, dict] = {},
        name: str = "VND",
    ):
        self.iterations = params.get("iters", 100)

        self.op_list = op_list
        self.perturb_op = OperatorMeta("Pick", op_list, {"init_idx": 0})

        self.nchange = (
            NeighborhoodChange.from_str(params["nchange"])
            if "nchange" in params
            else NeighborhoodChange.SEQ
        )

        self.current_op = 0

        if selection_op is None:
            selection_op = SurvivorSelection("One-to-One")
        self.selection_op = selection_op

        self.inner_selection_op = SurvivorSelection("One-to-One")

        if pop_init.pop_size > 1:
            pop_init.pop_size = 1
            warnings.warn(
                "The RVNS algorithm work on a single individual. The population size has been set to 1.",
                stacklevel=2,
            )

        super().__init__(pop_init, params=params, name=name)

    def perturb(self, indiv_list, objfunc, **kwargs):
        next_indiv_list = copy(indiv_list)
        for _ in range(self.iterations):
            offspring = []
            for indiv in indiv_list:
                # Perturb individual
                new_indiv = self.perturb_op(
                    indiv, indiv_list, objfunc, self.best, self.pop_init
                )
                new_indiv.genotype = objfunc.repair_solution(new_indiv.genotype)

                offspring.append(new_indiv)

            # Keep best individual regardless of selection method
            current_best = max(offspring, key=lambda x: x.fitness)
            if self.best.fitness < current_best.fitness:
                self.best = current_best

            next_indiv_list = self.inner_selection_op(next_indiv_list, offspring)

        return next_indiv_list

    def select_individuals(self, population, offspring, **kwargs):
        new_population = self.selection_op(population, offspring)

        self.perturb_op.chosen_idx = next_neighborhood(
            offspring[0], population[0], self.perturb_op.chosen_idx, self.nchange
        )

        return new_population

    def update_params(self, **kwargs):
        progress = kwargs["progress"]

        if isinstance(self.perturb_op, Operator):
            self.perturb_op.step(progress)

        if (
            self.perturb_op.chosen_idx >= len(self.op_list)
            or self.perturb_op.chosen_idx < 0
        ):
            self.perturb_op.chosen_idx = 0

    def extra_step_info(self):
        idx = self.perturb_op.chosen_idx

        print(
            f"\tCurrent Operator: {idx}/{len(self.op_list)}, {self.op_list[idx].name}"
        )
        # time.sleep(0.25)
