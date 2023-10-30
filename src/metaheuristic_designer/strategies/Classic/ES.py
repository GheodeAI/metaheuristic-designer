from __future__ import annotations
from typing import Union
from ...operators import OperatorMeta
from ..VariablePopulation import VariablePopulation


class ES(VariablePopulation):
    """
    Evolution strategy
    """

    def __init__(
        self,
        pop_init: Initializer,
        mutation_op: Operator,
        cross_op: Operator,
        parent_sel_op: ParentSelection,
        selection_op: SurvivorSelection,
        params: Union[ParamScheduler, dict] = {},
        name: str = "ES",
    ):
        offspring_size = params.get("offspringSize", pop_init.pop_size)
        evolve_op = OperatorMeta("Sequence", [mutation_op, cross_op])

        super().__init__(
            pop_init,
            evolve_op,
            parent_sel_op=parent_sel_op,
            selection_op=selection_op,
            n_offspring=offspring_size,
            params=params,
            name=name,
        )
