from __future__ import annotations
from ...initializer import Initializer
from ...operator import Operator
from ...selection_methods import ParentSelection, SurvivorSelection
from ...param_scheduler import ParamScheduler
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
            evolve_op = CompositeOperator([mutation_op, cross_op])

        offspring_size = params.get("offspringSize", initializer.pop_size)

        super().__init__(
            initializer,
            operator=evolve_op,
            parent_sel=parent_sel,
            survivor_sel=survivor_sel,
            n_offspring=offspring_size,
            params=params,
            name=name,
        )
