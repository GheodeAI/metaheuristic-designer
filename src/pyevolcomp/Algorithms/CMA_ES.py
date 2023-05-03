from __future__ import annotations
from .ES import ES
from ..Operators import OperatorReal, OperatorMeta
from ..SurvivorSelection import SurvivorSelection
from ..ParentSelection import ParentSelection
from ..Encodings import CMAEncoding


class CMA_ES(ES):
    def __init__(self, mutation_op: Operator, cross_op: Operator, parent_sel_op: ParentSelection, selection_op: SurvivorSelection,
                 params: Union[ParamScheduler, dict] = {}, name: str = "ES"):

        parent_select = ParentSelection("Nothing")
        selection = SurvivorSelection("(m+n)")

        cross = OperatorReal("Multicross", {"N": params["HMS"]})
        mutate1 = OperatorReal("MutNoise", {"method": "Gauss", "F": params["BW"], "Cr": params["HMCR"] * params["PAR"]})
        rand1 = OperatorReal("RandomMask", {"Cr": 1 - params["HMCR"]})

        mutate = OperatorMeta("Sequence", [mutate1, rand1])

        super().__init__(mutate, cross, parent_select, selection, params, name)

    def initialize(self, objfunc):
        objfunc.encoding = CMAEncoding(self.params["nparams"], pre_Encoding=objfunc.Encoding)
        super().initialize(objfunc)
