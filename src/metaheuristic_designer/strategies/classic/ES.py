from __future__ import annotations
from typing import Optional
from ...initializer import Initializer
from ...operator import Operator
from ...survivor_selection import SurvivorSelection
from ...parent_selection import ParentSelection
from ...operators import CompositeOperator
from ..variable_population import VariablePopulation


class ES(VariablePopulation):
    """
    Evolution strategy
    """

    def __init__(
        self,
        initializer: Initializer,
        mutation_op: Operator,
        cross_op: Optional[Operator] = None,
        parent_sel: Optional[ParentSelection] = None,
        survivor_sel: Optional[SurvivorSelection] = None,
        offspring_size: Optional[int] = None,
        name: str = "ES",
        **kwargs,
    ):
        if cross_op is None:
            evolve_op = mutation_op
        else:
            evolve_op = CompositeOperator([mutation_op, cross_op])

        super().__init__(
            initializer, operator=evolve_op, parent_sel=parent_sel, survivor_sel=survivor_sel, offspring_size=offspring_size, name=name, **kwargs
        )
