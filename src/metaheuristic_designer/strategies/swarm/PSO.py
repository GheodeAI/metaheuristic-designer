from __future__ import annotations
from ...initializer import Initializer, ExtendedInitializer
from ...operator import Operator
from ...operators import VectorOperator
from ..static_population import StaticPopulation
from ...param_scheduler import ParamScheduler


class PSO(StaticPopulation):
    """
    Particle swarm optimization
    """

    def __init__(
        self,
        initializer: ExtendedInitializer,
        params: ParamScheduler | dict = None,
        name: str = "PSO",
    ):
        assert isinstance(initializer, ExtendedInitializer), "Using the PSO strategy needs an `ExtendedInitializer` with a `speed` extension."

        if params is None:
            params = {}

        pso_op = VectorOperator(
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

        super().__init__(initializer, operator=pso_op, params=params, name=name)

    def extra_step_info(self):
        popul_matrix = self.population.genotype_matrix
        divesity = popul_matrix.std(axis=1).mean()
        # mean_speed = speed_matrix.mean()
        print(f"\tdiversity: {divesity:0.3}")
        # print(f"\tmean speed: {mean_speed:0.3}")

