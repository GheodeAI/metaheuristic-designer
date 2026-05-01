from __future__ import annotations
from typing import Optional
import numpy as np
from ...initializer import Initializer
from ...survivor_selection import create_survivor_selection
from ...operator import Operator
from ..hill_climb import HillClimb
from ...schedulable_parameter import SchedulableParameter
from ...utils import check_random_state, RNGLike


class SA(HillClimb):
    """
    Simulated annealing
    """

    def __init__(
        self,
        initializer: Initializer,
        operator: Operator,
        name: str = "SA",
        iterations: int | SchedulableParameter = 100,
        temperature_init: float | SchedulableParameter = 100,
        alpha: float | SchedulableParameter = 0.99,
        random_state: Optional[RNGLike] = None,
        **kwargs,
    ):

        # We need to do the check earlier since it will be injected into the survivor selection
        # and we want everything to share the random state if possible.
        random_state = check_random_state(random_state)

        # We can't access temperature_init yet, it could be a SchedulableParameter,
        # we fix the p after the constructor.
        survivor_sel = create_survivor_selection("probabilistic_hillclimb", p=None, random_state=random_state)

        self.iter_count = 0
        super().__init__(
            initializer,
            operator=operator,
            survivor_sel=survivor_sel,
            name=name,
            random_state=random_state,
            # Forced kwargs
            iterations=iterations,
            temperature_init=temperature_init,
            alpha=alpha,
            **kwargs,
        )

        self.temperature = self.params.temperature_init
        survivor_sel.update_kwargs(p=np.exp(-1 / self.temperature))

    def step(self, progress):
        super().step(progress=progress)

        self.iter_count += 1
        if self.iter_count > self.params.iterations:
            self.temperature *= self.params.alpha
            self.iter_count = 0
            self.survivor_sel.update_kwargs(p=np.exp(-1 / self.temperature))

    def extra_step_info(self):
        print()
        print(f"\tTemp iters: {self.iter_count}/{self.params.iterations}")
        print(f"\tTemperature: {self.temperature:0.4f}")
        print(f"\tAccept prob: {np.exp(-1 / self.temperature):0.4f}")
