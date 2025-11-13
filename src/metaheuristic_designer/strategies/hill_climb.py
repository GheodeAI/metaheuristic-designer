from __future__ import annotations
from ..initializer import Initializer
from ..param_scheduler import ParamScheduler
from ..search_strategy import SearchStrategy
from ..operator import Operator
from ..selection_methods import SurvivorSelection


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

        super().__init__(
            initializer,
            operator=operator,
            survivor_sel=survivor_sel,
            params=params,
            name=name,
        )

    def update_params(self, **kwargs):
        super().update_params(**kwargs)

        progress = kwargs["progress"]

        if isinstance(self.operator, Operator):
            self.operator.step(progress)
