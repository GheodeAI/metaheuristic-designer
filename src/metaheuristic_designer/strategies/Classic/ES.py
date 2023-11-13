from __future__ import annotations
from typing import Union
from ...operators import OperatorMeta, OperatorReal
from ...selectionMethods import ParentSelection
from ..VariablePopulation import VariablePopulation


class ES(VariablePopulation):
    """
    Evolution strategy
    """

    def __init__(
        self,
        initializer: Initializer,
        mutation_op: Operator,
        cross_op: Operator = None,
        parent_sel_op: ParentSelection = None,
        selection_op: SurvivorSelection = None,
        params: Union[ParamScheduler, dict] = {},
        name: str = "ES",
    ):
        if cross_op is None:
            evolve_op = mutation_op
        else:
            evolve_op = OperatorMeta("Sequence", [mutation_op, cross_op])

        offspring_size = params.get("offspringSize", initializer.pop_size)

        super().__init__(
            initializer,
            evolve_op,
            parent_sel_op=parent_sel_op,
            selection_op=selection_op,
            n_offspring=offspring_size,
            params=params,
            name=name,
        )
