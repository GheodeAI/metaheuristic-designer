from __future__ import annotations
import numpy as np
from ...initializers import UniformInitializer, ExtendedInitializer
from ...operators import create_swarm_operator
from ...encodings import ParameterExtendingEncoding
from ..static_population import StaticPopulation


class PSO(StaticPopulation):
    """
    Particle swarm optimization
    """

    def __init__(
        self,
        encoding: ParameterExtendingEncoding,
        initializer: ExtendedInitializer = None,
        population_size: int = 100,
        low_lim: float = -100,
        up_lim: float = 100,
        name: str = "PSO",
        w=0.7,
        c1=1.5,
        c2=1.5,
        **kwargs,
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

        pso_op = create_swarm_operator("PSO", encoding=encoding, w=w, c1=c1, c2=c2)

        super().__init__(initializer, operator=pso_op, name=name, **kwargs)
