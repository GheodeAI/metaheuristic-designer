from __future__ import annotations
from typing import Union
import numpy as np
from ...operators import OperatorReal, OperatorMeta
from ..VariablePopulation import VariablePopulation


class GA(VariablePopulation):
    """
    Genetic algorithm
    """

    def __init__(
        self,
        pop_init: Initializer,
        mutation_op: Operator,
        cross_op: Operator,
        parent_sel_op: ParentSelection,
        selection_op: SurvivorSelection,
        params: Union[ParamScheduler, dict] = {},
        name: str = "GA",
    ):
        self.pmut = params.get("pmut", 0.1)
        self.pcross = params.get("pcross", 0.9)

        null_operator = OperatorReal("Nothing")

        prob_mut_op = OperatorMeta(
            "Branch", [mutation_op, null_operator], {"p": self.pmut}
        )
        prob_cross_op = OperatorMeta(
            "Branch", [cross_op, null_operator], {"p": self.pcross}
        )

        evolve_op = OperatorMeta("Sequence", [prob_mut_op, prob_cross_op])

        super().__init__(
            pop_init,
            evolve_op,
            parent_sel_op=parent_sel_op,
            selection_op=selection_op,
            n_offspring=pop_init.pop_size,
            params=params,
            name=name,
        )

    def extra_step_info(self):
        popul_matrix = np.array(list(map(lambda x: x.genotype, self.population)))
        divesity = popul_matrix.std(axis=1).mean()
        print(f"\tdiversity: {divesity:0.3}")
