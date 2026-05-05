from __future__ import annotations
import logging
from typing import Optional
import numpy as np

from ...constraint_handlers.extended_constraint import ExtendedConstraintHandler
from ...encodings.composite_encoding import CompositeEncoding
from ...encodings.special.PSO_encoding import PSOEncoding
from ...initializer import Initializer
from ...constraint_handlers.bounce_bound_constraint import BounceBoundConstraint
from ...initializers import UniformInitializer, ExtendedInitializer
from ...operators import create_swarm_operator
from ...encodings import ParameterExtendingEncoding
from ..static_population import StaticPopulation
from ...utils import RNGLike

logger = logging.getLogger(__name__)


class PSO(StaticPopulation):
    """
    Particle swarm optimization
    """

    def __init__(
        self,
        initializer: Initializer,
        population_size: int = 100,
        lower_bound: float = -100,
        upper_bound: float = 100,
        name: str = "PSO",
        w=0.7,
        c1=1.5,
        c2=1.5,
        encoding: Optional[ParameterExtendingEncoding] = None,
        random_state: Optional[RNGLike] = None,
        **kwargs,
    ):
        if encoding is None:
            encoding = PSOEncoding(initializer.vecsize)
        elif not isinstance(encoding, ParameterExtendingEncoding):
            encoding = CompositeEncoding([PSOEncoding(initializer.vecsize), encoding])

        self.abs_upper_bound = np.maximum(np.abs(lower_bound), np.abs(upper_bound))
        if initializer is None:
            initializer = (UniformInitializer(encoding.vecsize, lower_bound, upper_bound, pop_size=population_size, random_state=random_state),)
        elif not isinstance(initializer.encoding, ParameterExtendingEncoding):
            logger.info("Overwritten initializer's encoding with PSO encoding.")
            initializer.encoding = encoding

        if not isinstance(initializer, ExtendedInitializer):
            initializer = ExtendedInitializer(
                solution_init=initializer,
                param_init_dict={
                    "speed": UniformInitializer(encoding.vecsize, -self.abs_upper_bound, self.abs_upper_bound, random_state=random_state)
                },
                random_state=random_state,
                encoding=encoding,
            )

        self.encoding = encoding

        pso_op = create_swarm_operator("PSO", encoding=encoding, w=w, c1=c1, c2=c2, random_state=random_state)

        super().__init__(initializer, operator=pso_op, name=name, **kwargs)

    def initialize(self, objfunc):
        if not isinstance(objfunc.constraint_handler, ExtendedConstraintHandler):
            objfunc.add_parameter_constraints(
                self.encoding, {"speed": BounceBoundConstraint(self.encoding.vecsize, -self.abs_upper_bound, self.abs_upper_bound)}
            )
        return super().initialize(objfunc)
