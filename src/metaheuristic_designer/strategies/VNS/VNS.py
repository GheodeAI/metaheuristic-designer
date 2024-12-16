from __future__ import annotations
from typing import Iterable
import warnings
from ...Population import Population
from ...Algorithm import Algorithm
from ...Initializer import Initializer
from ...ParamScheduler import ParamScheduler
from ...SearchStrategy import SearchStrategy
from ...Operator import Operator
from ...operators import OperatorMeta
from ...selectionMethods import SurvivorSelection, ParentSelectionNull
from .vns_neighborhood_changes import *


class VNS(SearchStrategy):
    """
    Variable neighborhood search
    """

    def __init__(
        self,
        initializer: Initializer,
        op_list: Iterable[Operator],
        local_search: Algorithm,
        survivor_sel: SurvivorSelection = None,
        params: ParamScheduler | dict = {},
        inner_loop_params: ParamScheduler | dict = {},
        name: str = "VNS",
    ):
        if params is None:
            params = {}

        if inner_loop_params is None:
            inner_loop_params = {}
        
        self.iterations = params.get("iters", 100)

        self.op_list = op_list
        operator = OperatorMeta("Pick", op_list, {"init_idx": 0})

        self.nchange = NeighborhoodChange.from_str(params["nchange"]) if "nchange" in params else NeighborhoodChange.SEQ

        self.local_search = local_search

        if initializer.pop_size > 1:
            initializer.pop_size = 1
            warnings.warn(
                "The VNS algorithm work on a single individual. The population size has been set to 1.",
                stacklevel=2,
            )

        super().__init__(
            initializer=initializer,
            operator=operator,
            survivor_sel=survivor_sel,
            params=params,
            name=name
        )

    def initialize(self, objfunc):
        initial_population = super().initialize(objfunc)
        initial_pop = self.local_search.initialize(objfunc)
        self.local_search.evaluate_population(initial_pop)

        return initial_population

    def perturb(self, parents, **kwargs):
        new_population = self.operator.evolve(parents, self.initializer)
        new_population = self.repair_population(new_population)

        # Local search
        self.local_search.operator = self.operator
        self.local_search.population = new_population
        for _ in range(self.iterations):
            parents_inner = self.local_search.select_parents(new_population)

            offspring_inner = self.local_search.perturb(parents_inner)
            offspring_inner = self.local_search.evaluate_population(offspring_inner)

            new_population = self.local_search.select_individuals(new_population, offspring_inner)

            self.population.update_best_from_parents(offspring_inner)

            self.local_search.update_params(**kwargs)

        return new_population

    def select_individuals(self, population, offspring, **kwargs):
        new_population = super().select_individuals(population, offspring, **kwargs)

        self.operator.chosen_idx = next_neighborhood(offspring.fitness[0], population.fitness[0], self.operator.chosen_idx, self.nchange)

        return new_population

    def update_params(self, **kwargs):
        super().update_params(**kwargs)

        progress = kwargs["progress"]

        if isinstance(self.operator, Operator):
            self.operator.step(progress)

        if self.operator.chosen_idx >= len(self.op_list) or self.operator.chosen_idx < 0:
            self.operator.chosen_idx = 0

    def extra_step_info(self):
        idx = self.operator.chosen_idx

        print(f"\tCurrent Operator: {idx}/{len(self.op_list)}, {self.op_list[idx].name}")
