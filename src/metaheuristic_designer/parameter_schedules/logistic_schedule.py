"""
Schedule that follows a sigmoidal (logistic) transition between two values.
"""

import scipy as sp
from ..schedulable_parameter import SchedulableParameter


class LogisticSchedule(SchedulableParameter):
    """
    Schedule that transitions between two values following a sigmoid curve.

    The steepness is controlled by `k`.  When `exact_bounds` is ``True``,
    the output is rescaled to exactly start at `init_value` and end at
    `final_value`.

    Parameters
    ----------
    init_value : float
        Starting value.
    final_value : float
        Target value.
    k : float, optional
        Steepness of the logistic curve (default 10).
    exact_bounds : bool, optional
        If ``True``, the output is rescaled to hit the exact bounds
        at progress 0 and 1.
    """

    def __init__(self, init_value: float, final_value:float , k: float = 10, exact_bounds: bool = False):
        super().__init__(random_state=None)
        self.init_value = init_value
        self.final_value = final_value
        self.k = k
        self.exact_bounds = exact_bounds

    def evaluate(self, progress: float) -> float:
        val_diff = self.final_value - self.init_value
        if self.exact_bounds:
            naive_logistic = sp.special.expit(self.k * (progress - 0.5))
            l_min = sp.special.expit(-self.k * 0.5)
            l_max = sp.special.expit(self.k * 0.5)
            return self.init_value + val_diff * (naive_logistic - l_min) / (l_max - l_min)
        else:
            return self.init_value + val_diff * sp.special.expit(self.k * (progress - 0.5))
