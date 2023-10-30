from __future__ import annotations
from typing import Union
from ...selectionMethods import SurvivorSelection
from ..StaticPopulation import StaticPopulation


class DE(StaticPopulation):
    """
    Differential evolution
    """

    def __init__(
        self,
        pop_init: Initializer,
        de_op: Operator,
        params: Union[ParamScheduler, dict] = {},
        selection_op: SurvivorSelection = None,
        name: str = "DE",
    ):
        if selection_op is None:
            selection_op = SurvivorSelection("One-to-one")

        super().__init__(
            pop_init, de_op, selection_op=selection_op, params=params, name=name
        )
