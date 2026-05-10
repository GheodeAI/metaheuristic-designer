"""
Schedule that changes value at discrete progress thresholds.
"""

from typing import Any

import numpy as np
from ..schedulable_parameter import SchedulableParameter

class StepSchedule(SchedulableParameter):
    """
    Schedule defined by a dictionary of progress-value pairs.

    At progress `p`, the schedule returns the value associated with the
    largest key ≤ `p`.  This produces a step function.

    Parameters
    ----------
    steps : dict
        Mapping of progress thresholds (floats in [0, 1]) to the values
        that should be active at or after that threshold.
    """

    def __init__(self, steps: dict[float, Any]):
        super().__init__(random_state=None)
        self.steps = steps

        self._cut_points = np.asarray(sorted(steps.keys()))
        self._output_values = np.asarray([self.steps[i] for i in self._cut_points])

    def evaluate(self, progress: float) -> float:
        idx = np.searchsorted(self._cut_points, progress, side="right") - 1
        idx = max(idx, 0)
        return self._output_values[idx]
