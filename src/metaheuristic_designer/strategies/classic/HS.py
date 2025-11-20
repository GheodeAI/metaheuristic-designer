from __future__ import annotations
from ...param_scheduler import ParamScheduler
from ...initializer import Initializer
from ...operators import VectorOperator, MetaOperator
from ...selection_methods import SurvivorSelection
from ..variable_population import VariablePopulation


class HS(VariablePopulation):
    """
    Harmony search
    """

    def __init__(
        self,
        initializer: Initializer,
        params: ParamScheduler | dict = None,
        name: str = "HS",
    ):
        if params is None:
            params = {}

        survivor_sel = SurvivorSelection("(m+n)")

        HSM = initializer.pop_size
        cross = VectorOperator("Multicross", {"Nindiv": HSM})

        mutate1 = VectorOperator(
            "MutNoise",
            {
                "distrib": "Gauss",
                "F": params["BW"],
                "Cr": params["HMCR"] * params["PAR"],
            },
        )
        rand1 = VectorOperator("RandomMask", {"Cr": 1 - params["HMCR"]})

        mutate = MetaOperator("Sequence", [mutate1, rand1])

        evolve_op = MetaOperator("Sequence", [cross, mutate])

        super().__init__(
            initializer,
            operator=evolve_op,
            survivor_sel=survivor_sel,
            n_offspring=1,
            params=params,
            name=name,
        )
