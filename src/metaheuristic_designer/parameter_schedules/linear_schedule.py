"""
Schedule that changes a value linearly between two endpoints.
"""

from ..schedulable_parameter import SchedulableParameter


class LinearSchedule(SchedulableParameter):
    """
    Schedule that interpolates linearly between `init_value` and `final_value`.

    Parameters
    ----------
    init_value : float
        Value at progress 0.
    final_value : float
        Value at progress 1.
    """

    def __init__(self, init_value: float, final_value: float):
        super().__init__(rng=None)
        self.init_value = init_value
        self.final_value = final_value

    def evaluate(self, progress: float) -> float:
        return (1 - progress) * self.init_value + progress * self.final_value
