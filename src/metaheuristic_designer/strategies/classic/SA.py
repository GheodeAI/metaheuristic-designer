"""
Simulated Annealing strategy.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
from ...initializer import Initializer
from ...survivor_selection import create_survivor_selection
from ...operator import Operator
from ...schedulable_parameter import SchedulableParameter
from ...utils import check_random_state, RNGLike
from ..single_solution_strategy import SingleSolutionStrategy
from ...parameter_schedules import ProbabilityAnnealingSchedule


class SA(SingleSolutionStrategy):
    """
    Simulated Annealing algorithm.

    A single solution is perturbed each iteration.  The new solution
    is accepted if it is better, or probabilistically if it is worse,
    according to an exponentially decaying temperature schedule.

    Parameters
    ----------
    initializer : Initializer
        Population initializer (usually creates a single individual).
    operator : Operator
        Perturbation operator.
    name : str, optional
        Display name (default ``"SA"``).
    iterations : int or SchedulableParameter, optional
        Number of iterations at constant temperature (default 100).
    temperature_init : float or SchedulableParameter, optional
        Starting temperature (default 100).
    alpha : float or SchedulableParameter, optional
        Cooling factor (default 0.99).
    random_state : RNGLike, optional
        Random number generator.
    **kwargs
        Forwarded to :class:`HillClimb`.
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
        p = ProbabilityAnnealingSchedule(temperature_init, iterations=iterations, alpha=alpha)

        super().__init__(
            initializer,
            operator=operator,
            survivor_sel=create_survivor_selection("probabilistic_hillclimb", p=p, random_state=random_state),
            name=name,
            random_state=random_state,
            **kwargs,
        )

    @property
    def temperature(self):
        return self.survivor_sel.raw_kwargs["p"].temperature

    def extra_step_info(self):
        """
        Displays temperature values and acceptance probability.
        """

        prob_schedule = self.survivor_sel.raw_kwargs["p"]

        print()
        print(f"\tTemp iters: {prob_schedule.iteration_counter}/{prob_schedule.iterations}")
        print(f"\tTemperature: {prob_schedule.temperature:0.4f}")
        print(f"\tAccept prob: {np.exp(-1 / prob_schedule.temperature):0.4f}")
