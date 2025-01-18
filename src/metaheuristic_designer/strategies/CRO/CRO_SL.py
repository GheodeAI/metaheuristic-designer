from __future__ import annotations
from typing import Union, List
from copy import deepcopy
from ...selectionMethods import SurvivorSelection
from ...ParamScheduler import ParamScheduler
from ...SearchStrategy import SearchStrategy
from ...Operator import Operator
from ...operators import OperatorMeta


class CRO_SL(SearchStrategy):
    """
    Coral Reef Optimization with Substrate Layers
    """

    def __init__(
        self,
        initializer: Initializer,
        operator_list: List[Operator],
        params: ParamScheduler | dict = None,
        name: str = "CRO-SL",
    ):
        initializer = deepcopy(initializer)
        initializer.pop_size = round(initializer.pop_size * params["rho"])

        super().__init__(initializer, params=params, name=name)

        # Hyperparameters of the algorithm
        self.maxpopsize = initializer.pop_size
        self.operator_list = operator_list
        self.operator_idx = [i % len(operator_list) for i in range(initializer.pop_size)]

        # self.operator = OperatorMeta("Branch", operator_list)
        self.operator = OperatorMeta("Pick", operator_list)
        self.operator.chosen_idx = self.operator_idx

        self.survivor_sel = SurvivorSelection(
            "CRO",
            {
                "Fd": params["Fd"],
                "Pd": params["Pd"],
                "attempts": params["attempts"],
                "maxPopSize": initializer.pop_size,
            },
        )

    def select_individuals(self, population, offspring, **kwargs):
        return self.survivor_sel(population, offspring)

    def update_params(self, **kwargs):
        super().update_params(**kwargs)

        progress = kwargs["progress"]

        self.initializer.pop_size = len(self.population)

        for op in self.operator_list:
            if isinstance(op, Operator):
                op.step(progress)

        if isinstance(self.params, ParamScheduler):
            self.params.step(progress)
