"""
Particle Swarm Optimization strategy.
"""

from __future__ import annotations
import logging
from typing import Optional
import numpy as np

from ...objective_function import ObjectiveFunc
from ...constraint_handlers.extended_constraint import ExtendedConstraintHandler
from ...encodings.composite_encoding import CompositeEncoding
from ...encodings.special.PSO_encoding import PSOEncoding
from ...initializer import Initializer
from ...constraint_handlers.bounce_bound_constraint import BounceBoundConstraint
from ...initializers import UniformInitializer, ExtendedInitializer
from ...operators import create_swarm_operator
from ...encodings import ParameterExtendingEncoding
from ..population_based_strategy import PopulationBasedStrategy
from ...utils import RNGLike

logger = logging.getLogger(__name__)


class PSO(PopulationBasedStrategy):
    """
    Particle Swarm Optimization (PSO).

    Each individual (particle) has a position and a velocity.  The
    velocity is updated using personal and global bests, and the
    position is moved accordingly.  This requires a
    :class:`ParameterExtendingEncoding` that stores a speed vector;
    if not supplied, a default :class:`PSOEncoding` is created.

    Parameters
    ----------
    initializer : Initializer
        Initializer for the solution part.  An
        :class:`ExtendedInitializer` is automatically created to
        handle the velocity parameter.
    lower_bound : float, optional
        Lower bound of the search space (default -100).
    upper_bound : float, optional
        Upper bound of the search space (default 100).
    name : str, optional
        Display name (default ``"PSO"``).
    w : float, optional
        Inertia weight (default 0.7).
    c1 : float, optional
        Cognitive acceleration coefficient (default 1.5).
    c2 : float, optional
        Social acceleration coefficient (default 1.5).
    encoding : ParameterExtendingEncoding, optional
        Encoding that includes a ``"speed"`` parameter.  If ``None``,
        a :class:`PSOEncoding` is used.
    random_state : RNGLike, optional
        Random number generator.
    **kwargs
        Forwarded to :class:`StaticPopulationStrategy`.
    """

    def __init__(
        self,
        initializer: Initializer,
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
            encoding = PSOEncoding(initializer.dimension)
        elif not isinstance(encoding, ParameterExtendingEncoding):
            encoding = CompositeEncoding([PSOEncoding(initializer.dimension), encoding])

        self.abs_upper_bound = np.maximum(np.abs(lower_bound), np.abs(upper_bound))
        if not isinstance(initializer.encoding, ParameterExtendingEncoding):
            logger.info("Overwritten initializer's encoding with PSO encoding.")
            initializer.encoding = encoding

        if not isinstance(initializer, ExtendedInitializer):
            initializer = ExtendedInitializer(
                solution_init=initializer,
                param_init_dict={
                    "speed": UniformInitializer(encoding.dimension, -self.abs_upper_bound, self.abs_upper_bound, random_state=random_state)
                },
                random_state=random_state,
                encoding=encoding,
            )

        self.encoding = encoding

        pso_op = create_swarm_operator("PSO", encoding=encoding, w=w, c1=c1, c2=c2, random_state=random_state)

        super().__init__(initializer, operator=pso_op, name=name, **kwargs)

    def initialize(self, objfunc: ObjectiveFunc):
        """Set up the initial population and attach velocity constraints.

        Parameters
        ----------
        objfunc : ObjectiveFunc
            The objective function.  Its constraint handler is extended
            with a :class:`BounceBoundConstraint` for the velocity so
            that speeds stay within the feasible range.

        .. warning::
        There is a known bug: the objective function **does not**
        automatically remove the extended constraint handler after
        a PSO run finishes.  Reusing the same objective function
        instance for other algorithms may cause unexpected
        behavior.  This will be resolved in a future release.

        Returns
        -------
        Population
            The initialized and evaluated population.
        """

        if not isinstance(objfunc.constraint_handler, ExtendedConstraintHandler):
            objfunc.add_parameter_constraints(
                self.encoding, {"speed": BounceBoundConstraint(self.encoding.dimension, -self.abs_upper_bound, self.abs_upper_bound)}
            )
        logger.info("Overwritten constraint's encoding with custom extended encoding. The objective must be reloaded for use with other algorithms.")
        return super().initialize(objfunc)
