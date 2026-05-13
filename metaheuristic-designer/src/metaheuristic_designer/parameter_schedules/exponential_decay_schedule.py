import logging
import numpy as np
from ..schedulable_parameter import SchedulableParameter

logger = logging.getLogger(__name__)


class ExponentialDecaySchedule(SchedulableParameter):
    def __init__(self, init_value, final_value=0, alpha=0.9, iterative=True):
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

    def evaluate(self, progress):
        if self.iterative:
            self.curr_value = self.final_value + (self.curr_value - self.final_value) * self.alpha
        else:
            self.curr_value = self.final_value + (self.init_value - self.final_value) * np.exp(-self.alpha * progress)
        return self.curr_value
