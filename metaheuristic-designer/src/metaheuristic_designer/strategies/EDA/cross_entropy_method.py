from typing import Optional
from ...initializer import Initializer
from ...parent_selection import create_parent_selection
from ...operators import create_mutation_operator
from ..static_population import StaticPopulation
from ...schedulable_parameter import SchedulableParameter
from ...utils import VectorLike, check_random_state, RNGLike


class CrossEntropyMethod(StaticPopulation):
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

    # TODO: add alpha smoothing for the mean each time the parent selection method is called.
