from __future__ import annotations
from ..StaticPopulation import StaticPopulation
from ...Operators import OperatorReal, OperatorMeta, OperatorNull
from ...SelectionMethods import SurvivorSelection, ParentSelection
from ...Encodings import CMAEncoding


class CMA_ES(StaticPopulation):
    def __init__(
        self,
        initializer: Initializer,
        parent_sel_op: ParentSelection,
        selection_op: SurvivorSelection,
        params: ParamScheduler | dict = {},
        name: str = "ES",
    ):
        parent_select = ParentSelection("Nothing")
        selection = SurvivorSelection("(m+n)")

        self.step = params.get("step", 1)

        self.generate_average = OperatorReal("Generate", {"statistic": "average"})
        self.sample_op = OperatorReal(
            "RandNoise",
            {"distrib": "MultiNormal", "mean": 0, "cov": [1], "F": self.step},
        )

        mutate = OperatorMeta("sequence", [self.generate_mean, self.sample_op])
        cross = OperatorNull()

        super().__init__(initializer, mutate, cross, parent_select, selection, params, name)

    def initialize(self, objfunc):
        objfunc.encoding = CMAEncoding(objfunc.vecsize**2, pre_encoding=objfunc.encoding)
        super().initialize(objfunc)

    def update_params(self, **kwargs):
        super().update_params(**kwargs)
