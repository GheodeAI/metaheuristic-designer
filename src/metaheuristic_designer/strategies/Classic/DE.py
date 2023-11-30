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
        initializer: Initializer,
        de_op: Operator,
        params: ParamScheduler | dict = {},
        survivor_sel: SurvivorSelection = None,
        name: str = "DE",
    ):
        if survivor_sel is None:
            survivor_sel = SurvivorSelection("One-to-one")

        super().__init__(initializer, de_op, survivor_sel=survivor_sel, params=params, name=name)
