from __future__ import annotations
from typing import Union
from copy import deepcopy
from ...operators import OperatorMeta
from ...selectionMethods import SurvivorSelection
from ..StaticPopulation import StaticPopulation
from ...ParamScheduler import ParamScheduler


class CRO(StaticPopulation):
    """
    Coral Reef Optimization
    """

    def __init__(
        self,
        initializer: Initializer,
        mutation_op: Operator,
        cross_op: Operator,
        params: ParamScheduler | dict = {},
        name: str = "CRO",
    ):
        evolve_op = OperatorMeta("Branch", [cross_op, mutation_op], {"p": params["Fb"]})

        survivor_sel = SurvivorSelection(
            "CRO",
            {
                "Fd": params["Fd"],
                "Pd": params["Pd"],
                "attempts": params["attempts"],
                "maxPopSize": initializer.pop_size,
            },
        )

        pop_init = deepcopy(initializer)
        pop_init.pop_size = round(pop_init.pop_size * params["rho"])

        super().__init__(pop_init, evolve_op, params=params, survivor_sel=survivor_sel, name=name)
