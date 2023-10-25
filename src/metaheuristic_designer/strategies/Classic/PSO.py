from __future__ import annotations
import numpy as np
from typing import Union
from ...operators import OperatorReal
from ...selectionMethods import SurvivorSelection
from ..StaticPopulation import StaticPopulation
from ...ParamScheduler import ParamScheduler


class PSO(StaticPopulation):
    """
    Particle swarm optimization
    """

    def __init__(
        self,
        pop_init: Initializer,
        params: Union[ParamScheduler, dict] = {},
        pso_op: Operator = None,
        name: str = "PSO",
    ):
        if pso_op is None:
            pso_op = OperatorReal(
                "PSO",
                ParamScheduler(
                    "Linear",
                    {
                        "w": params.get("w", 0.7),
                        "c1": params.get("c1", 1.5),
                        "c2": params.get("c2", 1.5),
                    },
                ),
            )

        selection_op = SurvivorSelection("Generational")

        super().__init__(
            pop_init, pso_op, selection_op=selection_op, params=params, name=name
        )

    def extra_step_info(self):
        popul_matrix = np.array(list(map(lambda x: x.genotype, self.population)))
        speed_matrix = np.array(list(map(lambda x: x.speed, self.population)))
        divesity = popul_matrix.std(axis=1).mean()
        mean_speed = speed_matrix.mean()
        print(f"\tdiversity: {divesity:0.3}")
        print(f"\tmean speed: {mean_speed:0.3}")
