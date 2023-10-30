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
        pop_init: Initializer,
        mutate: Operator,
        cross: Operator,
        params: Union[ParamScheduler, dict] = {},
        name: str = "CRO",
    ):
        evolve_op = OperatorMeta("Branch", [cross, mutate], {"p": params["Fb"]})

        selection_op = SurvivorSelection(
            "CRO",
            {
                "Fd": params["Fd"],
                "Pd": params["Pd"],
                "attempts": params["attempts"],
                "maxPopSize": pop_init.pop_size,
            },
        )

        pop_init = deepcopy(pop_init)
        pop_init.pop_size = round(pop_init.pop_size * params["rho"])

        super().__init__(
            pop_init, evolve_op, params=params, selection_op=selection_op, name=name
        )
