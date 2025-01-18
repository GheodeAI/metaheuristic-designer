from __future__ import annotations
from ..Initializer import Initializer
from ..ParamScheduler import ParamScheduler
from ..SearchStrategy import SearchStrategy
from ..Operator import Operator
from ..selectionMethods import SurvivorSelection


class HillClimb(SearchStrategy):
    """
    Hill Climbing algorithm
    """

    def __init__(
        self,
        initializer: Initializer,
        operator: Operator = None,
        survivor_sel: SurvivorSelection = None,
        params: ParamScheduler | dict = None,
        name: str = "HillClimb",
    ):
        if survivor_sel is None:
            survivor_sel = SurvivorSelection("HillClimb")

        super().__init__(initializer, operator=operator, survivor_sel=survivor_sel, params=params, name=name)

    def update_params(self, **kwargs):
        super().update_params(**kwargs)

        progress = kwargs["progress"]

        if isinstance(self.operator, Operator):
            self.operator.step(progress)
