"""
Annealing schedule for probabilities as seen in simulated annealing.
"""

import logging
import numpy as np
from .exponential_decay_schedule import ExponentialDecaySchedule
from .strided_schedule import StridedSchedule

logger = logging.getLogger(__name__)


class ProbabilityAnnealingSchedule(StridedSchedule):
    """Annealing strategies for probabilities.

    Holds a temperature parameter that is exponentially decayed and transformed into a probability
    using an ExponentialDecaySchedule internally.

    Parameters
    ----------
    temperature_init : int, optional
        Initial temperature, by default 100
    iterations : int, optional
        iterations to keep the previous parameter without updating, by default 100
    alpha : float, optional
        multiplier to apply to the temperature each update, by default 0.99
    rng
        Random state.
    """

    def __init__(self, temperature_init=100, iterations=100, alpha=0.99, rng=None):
        if alpha > 1 or alpha < 0:
            logger.warning(
                "It is HIGHLY recommended that `alpha` stays between 0 and 1 when using iterative exponential decay. Please ensure you know what you're doing"
            )

        subschedule = ExponentialDecaySchedule(init_value=temperature_init, alpha=alpha, iterative=True)
        self.temperature = temperature_init
        super().__init__(subschedule=subschedule, iterations=iterations)

    def evaluate(self, progress: float) -> float:
        self.temperature = super().evaluate(progress)
        return np.exp(-1 / self.temperature)
