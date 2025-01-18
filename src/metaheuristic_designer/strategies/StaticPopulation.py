from __future__ import annotations
from ..Initializer import Initializer
from ..ParamScheduler import ParamScheduler
from ..selectionMethods import SurvivorSelection,ParentSelection
from ..SearchStrategy import SearchStrategy
from ..Operator import Operator


class StaticPopulation(SearchStrategy):
    """
    Population-based algorithm where each individual is iteratively evolved with a given operator
    """

    def __init__(
        self,
        initializer: Initializer,
        operator: Operator,
        parent_sel: ParentSelection = None,
        survivor_sel: SurvivorSelection = None,
        params: ParamScheduler | dict = None,
        name: str = "Static Population Evolution",
    ):
        super().__init__(
            initializer,
            operator=operator,
            parent_sel=parent_sel,
            survivor_sel=survivor_sel,
            params=params,
            name=name,
        )

    def update_params(self, **kwargs):
        super().update_params(**kwargs)

        progress = kwargs["progress"]

        if isinstance(self.operator, Operator):
            self.operator.step(progress)

        if isinstance(self.survivor_sel, SurvivorSelection):
            self.survivor_sel.step(progress)

        if self.param_scheduler:
            self.param_scheduler.step(progress)
            self.params = self.param_scheduler.get_params()
