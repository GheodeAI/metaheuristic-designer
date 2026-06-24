"""Strided schedule that applies a subschedule making the parameter updates in long intervals."""

import logging
from ..schedulable_parameter import SchedulableParameter

logger = logging.getLogger(__name__)


class StridedSchedule(SchedulableParameter):
    """Schedule that applies a subschedule when a number of iterations have passed, keeping the previous
    value between updates.

    Parameters
    ----------
    subschedule: SchedulableParameter
        Parameter schedule to modify the parameter each `iterations` iterations.
    iterations : int, optional
        iterations to keep the current value unchanged, by default 100
    """

    def __init__(self, subschedule: SchedulableParameter, iterations: int = 100):
        super().__init__(rng=None)

        self.subschedule = subschedule
        self.iterations = iterations
        self.iteration_counter = 0
        self.current_value = None

    def evaluate(self, progress: float) -> float:
        if self.current_value is None or self.iteration_counter >= self.iterations - 1:
            self.current_value = self.subschedule.evaluate(progress)
            self.iteration_counter = 0
        else:
            self.iteration_counter += 1

        return self.current_value
