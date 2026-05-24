"""
Schedule that decays a value exponentially, either continuously or iteratively.
"""

import logging
import numpy as np
from ..schedulable_parameter import SchedulableParameter

logger = logging.getLogger(__name__)


class ExponentialDecaySchedule(SchedulableParameter):
    """
    Schedule that exponentially decays a value from `init_value` towards
    `final_value`.

    In iterative mode (`iterative=True`, the default), the current value
    is multiplied by `alpha` each time the schedule is evaluated.  In
    continuous mode, the decay follows the function
    ``final_value + (init_value - final_value) * exp(-alpha * progress)``.

    Parameters
    ----------
    init_value : float
        Starting value at progress 0.
    final_value : float, optional
        Asymptotic value (default 0).
    alpha : float, optional
        Decay factor.  In iterative mode it must be in (0, 1); in
        continuous mode it controls the rate of decay.
    iterative : bool, optional
        If ``True`` (default), the value is updated step-by-step.
        If ``False``, decay is computed directly from progress.
    """

    def __init__(self, init_value: float, final_value: float = 0, alpha: float = 0.9, iterative: bool = True):
        super().__init__(random_state=None)

        if iterative and (alpha > 1 or alpha < 0):
            logger.warning(
                "It is HIGHLY recommended that `alpha` stays between 0 and 1 when using iterative exponential decay. Please ensure you know what you're doing"
            )

        self.init_value = init_value
        self.final_value = final_value
        self.curr_value = init_value
        self.alpha = alpha
        self.iterative = iterative

    def evaluate(self, progress: float) -> float:
        if progress != 0:
            if self.iterative:
                self.curr_value = self.final_value + (self.curr_value - self.final_value) * self.alpha
            else:
                self.curr_value = self.final_value + (self.init_value - self.final_value) * np.exp(-self.alpha * progress)
        return self.curr_value
