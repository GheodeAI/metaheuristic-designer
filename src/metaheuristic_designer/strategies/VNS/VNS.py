from __future__ import annotations
from typing import Union
import time
import warnings
from ...algorithms import GeneralAlgorithm
from ...ParamScheduler import ParamScheduler
from ...SearchStrategy import SearchStrategy
from ...Operator import Operator
from ...operators import OperatorMeta
from ...selectionMethods import SurvivorSelection, ParentSelectionNull
from .VND import VND
from .vns_neighborhood_changes import *


class VNS(SearchStrategy):
    """
    Variable neighborhood search
    """

    def __init__(
        self,
        initializer: Initializer,
        op_list: List[Operator],
        local_search_strategy: SearchStrategy = None,
        survivor_sel: SurvivorSelection = None,
        params: ParamScheduler | dict = {},
        inner_loop_params: ParamScheduler | dict = {},
        name: str = "VNS",
    ):
        self.op_list = op_list
        operator = OperatorMeta("Pick", op_list, {"init_idx": 0})

        self.nchange = NeighborhoodChange.from_str(params["nchange"]) if "nchange" in params else NeighborhoodChange.SEQ

        if local_search_strategy is None:
            local_search_strategy = VND(initializer=initializer, op_list=op_list, one_shot=True)
        local_search_strategy.name = f"VNS ({local_search_strategy.name})"

        inner_loop_params['init_info'] = False

        self.local_search = GeneralAlgorithm(
            objfunc=None,
            search_strategy=local_search_strategy,
            params=inner_loop_params,
            name=local_search_strategy.name
        )

        if survivor_sel is None:
            survivor_sel = SurvivorSelection("One-to-One")

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
        
        self.local_search.objfunc = objfunc
        self.local_search.initialize()

        return initial_population

    def perturb(self, indiv_list, objfunc, **kwargs):
        new_population = self.operator(indiv_list, objfunc, self.best, self.initializer)
        new_population = self.repair_population(new_population, objfunc)

        # Local search
        self.local_search.search_strategy.finish = False
        self.local_search.restart(reset_objfunc=False)
        self.local_search.search_strategy.population = new_population
        self.local_search.search_strategy.best = new_population[0]
        self.local_search.optimize(initialize=False)
        
        offspring = self.local_search.search_strategy.population

        return offspring

    def select_individuals(self, population, offspring, **kwargs):
        next_population = super().select_individuals(population, offspring, **kwargs)

        self.operator.chosen_idx = next_neighborhood(offspring[0], population[0], self.operator.chosen_idx, self.nchange)

        return next_population

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
