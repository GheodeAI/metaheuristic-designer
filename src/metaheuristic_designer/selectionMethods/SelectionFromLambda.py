from __future__ import annotations
from ..ParamScheduler import ParamScheduler
from ..SelectionMethod import SelectionMethod


class SelectionFromLambda(SelectionMethod):
    def __init__(
        self,
        select_fn: callable,
        params: ParamScheduler | dict,
        padding: bool = False,
        name: str = None,
    ):
        if name is None:
            name = select_fn.__name__

        super().__init__(params, padding, name)

    def select(self, popul: List[Individual], offspring: List[Individual] = None) -> List[Individual]:
        return select_fn(popul, offspring, **self.params)
