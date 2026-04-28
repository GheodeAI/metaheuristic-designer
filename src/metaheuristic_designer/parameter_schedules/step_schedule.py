import numpy as np
from ..schedulable_parameter import SchedulableParameter


class StepSchedule(SchedulableParameter):
    def __init__(self, steps):
        super().__init__(random_state=None)
        self.steps = steps

        self._cut_points = np.asarray(sorted(steps.keys()))
        self._output_values = np.asarray([self.steps[i] for i in self._cut_points])

    def evaluate(self, progress):
        idx = np.searchsorted(self._cut_points, progress, side="right") - 1
        idx = max(idx, 0)
        return self._output_values[idx]
