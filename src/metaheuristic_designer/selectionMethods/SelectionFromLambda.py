from __future__ import annotations
from ..Population import Population
from ..ParamScheduler import ParamScheduler
from ..SelectionMethod import SelectionMethod


class SelectionFromLambda(SelectionMethod):
    def __init__(
        self,
        select_fn: callable,
        params: ParamScheduler | dict,
        name: str = None,
    ):
        if name is None:
            name = select_fn.__name__

        self.select_fn = select_fn

        super().__init__(params, name)

    def select(self, population: Population, offspring: Population = None) -> Population:
        return self.select_fn(population, offspring, **self.params)
