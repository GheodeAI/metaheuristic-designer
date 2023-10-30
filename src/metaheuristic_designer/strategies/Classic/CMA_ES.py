from __future__ import annotations
from ..StaticPopulation import StaticPopulation
from ...Operators import OperatorReal, OperatorMeta
from ...SelectionMethods import SurvivorSelection, ParentSelection
from ...Encodings import CMAEncoding


class CMA_ES(StaticPopulation):
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
        parent_select = ParentSelection("Nothing")
        selection = SurvivorSelection("(m+n)")

        cross = OperatorReal("Multicross", {"N": params["HMS"]})
        mutate1 = OperatorReal(
            "MutNoise",
            {
                "method": "Gauss",
                "F": params["BW"],
                "Cr": params["HMCR"] * params["PAR"],
            },
        )
        rand1 = OperatorReal("RandomMask", {"Cr": 1 - params["HMCR"]})

        mutate = OperatorMeta("Sequence", [mutate1, rand1])

        super().__init__(
            pop_init, mutate, cross, parent_select, selection, params, name
        )

    def initialize(self, objfunc):
        objfunc.encoding = CMAEncoding(
            self.params["nparams"], pre_encoding=objfunc.encoding
        )
        super().initialize(objfunc)
