"""
Schedule that switches between two values at a fixed progress threshold.
"""

from ..schedulable_parameter import SchedulableParameter


class ThresholdSchedule(SchedulableParameter):
    """
    Schedule that returns `init_value` until `progress` reaches `threshold`,
    then switches to `final_value`.

    Parameters
    ----------
    init_value : float
        Value used before the threshold.
    final_value : float
        Value used after the threshold.
    threshold : float, optional
        Progress point at which the switch occurs (default 0.5).
    """

    def __init__(self, init_value: float, final_value: float, threshold: float = 0.5):
        super().__init__(rng=None)
        self.init_value = init_value
        self.final_value = final_value
        self.threshold = threshold

    def evaluate(self, progress: float) -> float:
        return self.init_value if progress < self.threshold else self.final_value
