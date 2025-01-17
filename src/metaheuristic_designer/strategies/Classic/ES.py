from __future__ import annotations
from ...Initializer import Initializer
from ...Operator import Operator
from ...selectionMethods import ParentSelection, SurvivorSelection
from ...ParamScheduler import ParamScheduler
from ...operators import OperatorMeta
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
        parent_sel: ParentSelection = None,
        survivor_sel: SurvivorSelection = None,
        params: ParamScheduler | dict = None,
        name: str = "ES",
    ):
        if params is None:
            params = {}

        if cross_op is None:
            evolve_op = mutation_op
        else:
            evolve_op = OperatorMeta("Sequence", [mutation_op, cross_op])

        offspring_size = params.get("offspringSize", initializer.pop_size)

        super().__init__(
            initializer,
            evolve_op,
            parent_sel=parent_sel,
            survivor_sel=survivor_sel,
            n_offspring=offspring_size,
            params=params,
            name=name,
        )
