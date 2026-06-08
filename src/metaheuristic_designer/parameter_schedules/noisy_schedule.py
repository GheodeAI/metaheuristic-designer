"""
Schedule that applies gaussian noise to a subschedule.
"""

import logging
from typing import Optional
from ..schedulable_parameter import SchedulableParameter
from ..utils import RNGLike

logger = logging.getLogger(__name__)


class NoisySchedule(SchedulableParameter):
    """Schedule that applies gaussian noise to a subschedule.

    Parameters
    ----------
    subschedule: SchedulableParameter
        Parameter schedule to modify the parameter each `iterations` iterations.
    noise_level : float, optional
        Standard deviation of the gaussian noise applied to the parameter value
    """

    def __init__(self, subschedule: SchedulableParameter, noise_level: float = 1e-2, rng: Optional[RNGLike] = None):
        super().__init__(rng=rng)

        self.subschedule = subschedule
        self.noise_level = noise_level

    def evaluate(self, progress: float) -> float:
        next_value = self.subschedule.evaluate(progress)
        return next_value + self.rng.normal(0, self.noise_level)
