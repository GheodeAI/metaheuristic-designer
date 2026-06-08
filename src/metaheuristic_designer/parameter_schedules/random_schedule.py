"""Schedule that picks a random value at each evaluation."""

from typing import Optional

from ..utils import RNGLike
from ..schedulable_parameter import SchedulableParameter


class RandomSchedule(SchedulableParameter):
    """Schedule that returns a uniform random value between `init_value` and
    `final_value` at every call, ignoring progress.

    Parameters
    ----------
    init_value : float
        Lower bound of the random interval.
    final_value : float
        Upper bound of the random interval.
    rng : RNGLike, optional
        Random number generator.
    """

    def __init__(self, init_value: float, final_value: float, rng: Optional[RNGLike] = None):
        super().__init__(rng=rng)
        self.init_value = init_value
        self.final_value = final_value

    def evaluate(self, progress: float) -> float:
        return self.rng.uniform(self.init_value, self.final_value)
