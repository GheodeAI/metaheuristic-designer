from __future__ import annotations
from typing import Optional
from ...initializer import Initializer
from ..static_population import StaticPopulation
from ...operators import create_differential_evolution_operator
from ...survivor_selection import SurvivorSelection, create_survivor_selection
from ...schedulable_parameter import SchedulableParameter
from ...utils import check_random_state, RNGLike


class DE(StaticPopulation):
    """
    Differential evolution
    """

    def __init__(
        self,
        initializer: Initializer,
        de_operator_name: str = "DE/best/1",
        survivor_sel: Optional[SurvivorSelection] = None,
        name: str = "DE",
        random_state: Optional[RNGLike] = None,
        F: float | SchedulableParameter = 0.8,
        Cr: float | SchedulableParameter = 0.9,
        p: float | SchedulableParameter = 0.1,
        **kwargs,
    ):
        # We need to do the check earlier since it will be injected into the operator
        # and we want everything to share the random state if possible.
        random_state = check_random_state(random_state)

        if survivor_sel is None:
            survivor_sel = create_survivor_selection("one_to_one")

        super().__init__(
            initializer,
            operator=create_differential_evolution_operator(de_operator_name, random_state=random_state, F=F, Cr=Cr, p=p),
            survivor_sel=survivor_sel,
            name=name,
            random_state=random_state,
            **kwargs,
        )
