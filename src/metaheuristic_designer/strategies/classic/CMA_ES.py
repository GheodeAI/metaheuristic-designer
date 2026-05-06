from __future__ import annotations
from ...initializer import Initializer
from ..static_population import StaticPopulation
from ...survivor_selection_base import SurvivorSelection
from ...parent_selection_base import ParentSelection


class CMA_ES(StaticPopulation):
    def __init__(self, initializer: Initializer, parent_sel_op: ParentSelection, selection_op: SurvivorSelection, name: str = "ES"):
        raise NotImplementedError()
        # parent_select = ParentSelection("Nothing")
        # selection = SurvivorSelection("(m+n)")

        # self.step = params.get("step", 1)

        # self.generate_average = VectorOperator("Generate", {"statistic": "average"})
        # self.sample_op = VectorOperator(
        #     "RandNoise",
        #     {"distrib": "MultiNormal", "mean": 0, "cov": [1], "F": self.step},
        # )

        # mutate = CompositeOperator([self.generate_average, self.sample_op])
        # cross = NullOperator()

        # super().__init__(initializer, mutate, cross, parent_select, selection, params, name)

    # def initialize(self, objfunc):
        # objfunc.encoding = CMAEncoding(objfunc.vecsize**2, pre_encoding=objfunc.encoding)
        # super().initialize(objfunc)

    # def update_params(self, **kwargs):
    #     super().update_params(**kwargs)
