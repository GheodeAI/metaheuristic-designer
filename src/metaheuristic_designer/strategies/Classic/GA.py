from __future__ import annotations
from ...Initializer import Initializer
from ...Operator import Operator
from ...selectionMethods import ParentSelection, SurvivorSelection
from ...ParamScheduler import ParamScheduler
from ...operators import OperatorMeta, OperatorNull
from ..VariablePopulation import VariablePopulation


class GA(VariablePopulation):
    """
    Genetic algorithm
    """

    def __init__(
        self,
        initializer: Initializer,
        mutation_op: Operator,
        cross_op: Operator,
        parent_sel: ParentSelection,
        survivor_sel: SurvivorSelection,
        params: ParamScheduler | dict = None,
        name: str = "GA",
    ):
        if params is None:
            params = {}

        self.pmut = params.get("pmut", 0.1)
        self.pcross = params.get("pcross", 0.9)

        null_operator = OperatorNull()

        prob_mut_op = OperatorMeta("Branch", [mutation_op, null_operator], {"p": self.pmut})
        prob_cross_op = OperatorMeta("Branch", [cross_op, null_operator], {"p": self.pcross})

        evolve_op = OperatorMeta("Sequence", [prob_cross_op, prob_mut_op])

        super().__init__(
            initializer,
            operator=evolve_op,
            parent_sel=parent_sel,
            survivor_sel=survivor_sel,
            params=params,
            name=name,
        )

    def extra_step_info(self):
        # popul_matrix = np.array(list(map(lambda x: x.genotype, self.population)))
        divesity = self.population.genotype_set.std(axis=1).mean()
        print(f"\tdiversity: {divesity:0.3}")
