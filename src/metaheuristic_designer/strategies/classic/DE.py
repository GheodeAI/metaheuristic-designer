from __future__ import annotations
from ...initializer import Initializer
from ...operator import Operator
from ...param_scheduler import ParamScheduler
from ...selection_methods import SurvivorSelection
from ..static_population import StaticPopulation


class DE(StaticPopulation):
    """
    Differential evolution
    """

    def __init__(
        self,
        initializer: Initializer,
        de_operator: Operator,
        params: ParamScheduler | dict = None,
        survivor_sel: SurvivorSelection = None,
        name: str = "DE",
    ):
        if params is None:
            params = {}

        if survivor_sel is None:
            survivor_sel = SurvivorSelection("One-to-one")

        super().__init__(
            initializer,
            operator=de_operator,
            survivor_sel=survivor_sel,
            params=params,
            name=name,
        )
