from __future__ import annotations
from ...Initializer import Initializer
from ...Operator import Operator
from ...operators import OperatorVector
from ..StaticPopulation import StaticPopulation
from ...ParamScheduler import ParamScheduler


class PSO(StaticPopulation):
    """
    Particle swarm optimization
    """

    def __init__(
        self,
        initializer: Initializer,
        params: ParamScheduler | dict = None,
        pso_op: Operator = None,
        name: str = "PSO",
    ):
        if params is None:
            params = {}

        if pso_op is None:
            pso_op = OperatorVector(
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

        super().__init__(initializer, pso_op, params=params, name=name)

    def extra_step_info(self):
        popul_matrix = self.population.genotype_set
        speed_matrix = self.population.speed_set
        divesity = popul_matrix.std(axis=1).mean()
        mean_speed = speed_matrix.mean()
        print(f"\tdiversity: {divesity:0.3}")
        print(f"\tmean speed: {mean_speed:0.3}")
