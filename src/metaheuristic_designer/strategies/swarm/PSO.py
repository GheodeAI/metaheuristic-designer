from __future__ import annotations
import numpy as np
from ...initializer import Initializer, ExtendedInitializer
from ...initializers import UniformInitializer
from ...operator import Operator
from ...operators import SwarmOperator
from ..static_population import StaticPopulation
from ...param_scheduler import ParamScheduler


class PSO(StaticPopulation):
    """
    Particle swarm optimization
    """

    def __init__(
        self,
        encoding: ExtendedEncoding,
        initializer: ExtendedInitializer = None,
        population_size: int = 100,
        low_lim: float = -100,
        up_lim: float = 100,
        params: ParamScheduler | dict = None,
        name: str = "PSO",
    ):
        if initializer is None:
            abs_up_lim = np.maximum(np.abs(low_lim), np.abs(up_lim))

            initializer = ExtendedInitializer(
                solution_init=UniformInitializer(encoding.vecsize, low_lim, up_lim, pop_size=population_size),
                param_init_dict={"speed": UniformInitializer(encoding.vecsize, -abs_up_lim, abs_up_lim)},
                encoding=encoding,
            )
        else:
            assert isinstance(initializer, ExtendedInitializer), "Using the PSO strategy needs an `ExtendedInitializer` with a `speed` extension."
            assert "speed" in initializer.param_init_dict

        if params is None:
            params = {}

        pso_op = SwarmOperator(
            "PSO",
            ParamScheduler(
                "Linear",
                {
                    "w": params.get("w", 0.7),
                    "c1": params.get("c1", 1.5),
                    "c2": params.get("c2", 1.5),
                },
            ),
            encoding=encoding
        )

        super().__init__(initializer, operator=pso_op, params=params, name=name)

    def extra_step_info(self):
        popul_matrix = self.population.genotype_matrix
        divesity = popul_matrix.std(axis=1).mean()
        # mean_speed = speed_matrix.mean()
        print(f"\tdiversity: {divesity:0.3}")
        # print(f"\tmean speed: {mean_speed:0.3}")

