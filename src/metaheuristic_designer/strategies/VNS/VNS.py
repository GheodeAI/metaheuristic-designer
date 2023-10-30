from __future__ import annotations
from typing import Union
import warnings
from ...ParamScheduler import ParamScheduler
from ...SearchStrategy import SearchStrategy
from ...Operator import Operator
from ...operators import OperatorMeta
from ...selectionMethods import SurvivorSelection
from .vns_neighborhood_changes import *


class VNS(SearchStrategy):
    """
    Variable neighborhood search
    """

    def __init__(
        self,
        pop_init: Initializer,
        op_list: List[Operator],
        local_search: Algorithm,
        selection_op: SurvivorSelection = None,
        params: Union[ParamScheduler, dict] = {},
        name: str = "VNS",
    ):
        self.iterations = params.get("iters", 100)

        self.op_list = op_list
        self.perturb_op = OperatorMeta("Pick", op_list, {"init_idx": 0})

        self.nchange = (
            NeighborhoodChange.from_str(params["nchange"])
            if "nchange" in params
            else NeighborhoodChange.SEQ
        )

        self.local_search = local_search

        if selection_op is None:
            selection_op = SurvivorSelection("One-to-One")
        self.selection_op = selection_op

        if pop_init.pop_size > 1:
            pop_init.pop_size = 1
            warnings.warn(
                "The VNS algorithm work on a single individual. The population size has been set to 1.",
                stacklevel=2,
            )

        super().__init__(pop_init, params=params, name=name)

    def initialize(self, objfunc):
        super().initialize(objfunc)
        self.local_search.initialize(objfunc)

    def perturb(self, indiv_list, objfunc, **kwargs):
        offspring = []
        for indiv in indiv_list:
            # Perturb individual
            new_indiv = self.perturb_op(
                indiv, indiv_list, objfunc, self.best, self.pop_init
            )
            new_indiv.genotype = objfunc.repair_solution(new_indiv.genotype)

            # Local search
            population = [new_indiv]
            self.local_search.perturb_op = self.perturb_op
            for _ in range(self.iterations):
                parents, _ = self.local_search.select_parents(population, kwargs)

                offspring = self.local_search.perturb(parents, objfunc, kwargs)

                population = self.local_search.select_individuals(
                    population, offspring, kwargs
                )

                self.local_search.update_params()

            new_indiv = self.local_search.population[0]

            offspring.append(new_indiv)

        # Keep best individual regardless of selection method
        current_best = max(offspring, key=lambda x: x.fitness)
        if self.best.fitness < current_best.fitness:
            self.best = current_best

        return offspring

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
