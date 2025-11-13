from __future__ import annotations
from ..initializer import Initializer
from ..param_scheduler import ParamScheduler
from ..search_strategy import SearchStrategy
from ..operator import Operator
from ..selection_methods import SurvivorSelection


class LocalSearch(SearchStrategy):
    """
    Local search algorithm
    """

    def __init__(
        self,
        initializer: Initializer,
        operator: Operator = None,
        survivor_sel: SurvivorSelection = None,
        params: ParamScheduler | dict = None,
        name: str = "LocalSearch",
    ):
        if params is None:
            params = {}

        if survivor_sel is None:
            survivor_sel = SurvivorSelection("Many-to-one")

        super().__init__(
            initializer,
            operator=operator,
            survivor_sel=survivor_sel,
            params=params,
            name=name,
        )

        self.iterations = params.get("iters", 100)

    def perturb(self, parents, **kwargs):
        new_population = parents.repeat(self.iterations)
        return super().perturb(new_population, **kwargs)

    def update_params(self, **kwargs):
        super().update_params(**kwargs)

        progress = kwargs["progress"]

        if isinstance(self.operator, Operator):
            self.operator.step(progress)
