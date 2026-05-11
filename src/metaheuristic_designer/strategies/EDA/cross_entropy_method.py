"""
Cross-Entropy Method (CEM) strategy.
"""

from typing import Optional

from ...population import Population
from ...initializer import Initializer
from ...parent_selection import create_parent_selection
from ...operators import create_mutation_operator
from ...schedulable_parameter import SchedulableParameter
from ...utils import VectorLike, check_random_state, RNGLike
from ..eda_strategy import EDAStrategy


class CrossEntropyMethod(EDAStrategy):
    """
    Cross-Entropy Method for continuous optimisation.

    At each generation, the best individuals are selected and the
    mean of a Gaussian distribution is updated to their location,
    optionally with a scale estimated from the data.  New solutions
    are sampled from this distribution.

    .. note::
       This class will be refactored when the EDA interface is
       finalised.  Smoothing (learning rate) for the mean still
       needs to be added.

    Parameters
    ----------
    initializer : Initializer
        Population initializer.
    name : str, optional
        Display name (default ``"CrossEntropyMethod"``).
    random_state : RNGLike, optional
        Random number generator.
    elite_amount : int or SchedulableParameter, optional
        Number of best individuals used to estimate the distribution.
    scale : VectorLike or ``"calculated"``, optional
        Standard deviation of the Gaussian.  If ``"calculated"``,
        it is estimated from the selected individuals.
    **kwargs
        Forwarded to :class:`StaticPopulation`.
    """

    def __init__(
        self,
        initializer: Initializer,
        name: str = "CrossEntropyMethod",
        random_state: Optional[RNGLike] = None,
        elite_amount: Optional[int | SchedulableParameter] = None,
        scale: VectorLike | str = "calculated",
        **kwargs,
    ):
        random_state = check_random_state(random_state)

        operator = create_mutation_operator("RandSample", distribution="Normal", loc="calculated", scale=scale, random_state=random_state)
        parent_sel = create_parent_selection("best", amount=elite_amount)

        super().__init__(initializer=initializer, operator=operator, parent_sel=parent_sel, name=name, **kwargs)
    
    def estimate_parameters(self, population: Population) -> Population:
        # TODO: add alpha smoothing for the mean each time the parent selection method is called.

        return self.operator

