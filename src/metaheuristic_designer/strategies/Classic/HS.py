from __future__ import annotations
from typing import Union
from ...operators import OperatorReal, OperatorMeta
from ...selectionMethods import SurvivorSelection, ParentSelection
from ..VariablePopulation import VariablePopulation


class HS(VariablePopulation):
    """
    Harmony search
    """

    def __init__(
        self,
        pop_init: Initializer,
        params: Union[ParamScheduler, dict] = {},
        name: str = "HS",
    ):
        parent_sel_op = ParentSelection("Nothing")
        selection_op = SurvivorSelection("(m+n)")

        HSM = pop_init.pop_size
        cross = OperatorReal("Multicross", {"Nindiv": HSM})

        mutate1 = OperatorReal(
            "MutNoise",
            {
                "method": "Gauss",
                "F": params["BW"],
                "Cr": params["HMCR"] * params["PAR"],
            },
        )
        rand1 = OperatorReal("RandomMask", {"Cr": 1 - params["HMCR"]})

        mutate = OperatorMeta("Sequence", [mutate1, rand1])

        evolve_op = OperatorMeta("Sequence", [cross, mutate])

        super().__init__(
            pop_init,
            evolve_op,
            parent_sel_op=parent_sel_op,
            selection_op=selection_op,
            n_offspring=1,
            params=params,
            name=name,
        )
