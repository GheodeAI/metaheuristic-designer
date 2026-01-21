from __future__ import annotations
from typing import Union
from copy import deepcopy
from ...operators import BranchOperator
from ...selection_methods import SurvivorSelection
from ..static_population import StaticPopulation
from ...param_scheduler import ParamScheduler


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
        evolve_op = BranchOperator([cross_op, mutation_op], params={"p": params["Fb"]})

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
