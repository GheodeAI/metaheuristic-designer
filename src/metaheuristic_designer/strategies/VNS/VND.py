from __future__ import annotations
from typing import Iterable
from copy import copy
import warnings
from ...Initializer import Initializer
from ...ParamScheduler import ParamScheduler
from ...SearchStrategy import SearchStrategy
from ...Operator import Operator
from ...operators import OperatorMeta
from ...selectionMethods import (
    SurvivorSelection,
    ParentSelectionNull,
)
from .vns_neighborhood_changes import *


class VND(SearchStrategy):
    """
    Variable Neighborhood Descent

    As seen in:
        Hansen, P., & Mladenovic, N. (2003). A tutorial on variable neighborhood search. Les Cahiers du GERAD ISSN, 711, 2440.
    """

    def __init__(
        self,
        initializer: Initializer,
        op_list: Iterable[Operator],
        survivor_sel: SurvivorSelection = None,
        one_shot: bool = False,
        params: ParamScheduler | dict = None,
        name: str = "VND",
    ):

        if params is None:
            params = {}

        self.op_list = op_list
        perturb_op = OperatorMeta("Pick", op_list, {"init_idx": 0})

        self.nchange = NeighborhoodChange.from_str(params["nchange"]) if "nchange" in params else NeighborhoodChange.SEQ
        self.new_loop_flag = False
        self.one_shot = one_shot

        self.current_op = 0

        if survivor_sel is None:
            survivor_sel = SurvivorSelection("One-to-One")

        if initializer.pop_size > 1:
            initializer.pop_size = 1
            warnings.warn(
                "The RVNS algorithm work on a single individual. The population size has been set to 1.",
                stacklevel=2,
            )

        super().__init__(
            initializer,
            operator=operator,
            survivor_sel=survivor_sel,
            params=params,
            name=name
        )

    def perturb(self, parents, **kwargs):
        next_parents = copy(parents)
        for _ in range(self.iterations):
            offspring = self.operator.evolve(parents, self.initializer)
            offspring = self.repair_population(offspring)
            offspring.calculate_fitness()

            # Keep best individual regardless of selection method
            self.population.update(offspring)

            next_parents = self.inner_selection_op(next_parents, offspring)

        return next_parents

    def select_individuals(self, population, offspring, **kwargs):
        new_population = super().select_individuals(population, offspring, **kwargs)

        new_chosen_idx = next_neighborhood(new_population[0], population[0], self.operator.chosen_idx, self.nchange)
        self.operator.chosen_idx = new_chosen_idx % len(self.op_list)

        self.new_loop_flag = self.new_loop_flag or new_chosen_idx >= len(self.op_list)
        if self.new_loop_flag:
            self.finish = self.one_shot

        return new_population

    def update_params(self, **kwargs):
        super().update_params(**kwargs)

        progress = kwargs["progress"]

        if isinstance(self.operator, Operator):
            self.operator.step(progress)

    def extra_step_info(self):
        idx = self.operator.chosen_idx

        if self.new_loop_flag:
            print(f"\tStarted new loop")
            self.new_loop_flag = False
        
        print(f"\tCurrent Operator: {idx}/{len(self.op_list)}, {self.op_list[idx].name}")
