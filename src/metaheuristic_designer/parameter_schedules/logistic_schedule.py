import scipy as sp
from ..schedulable_parameter import SchedulableParameter


class LogisticSchedule(SchedulableParameter):
    def __init__(self, init_value, final_value, k=10, exact_bounds=False):
        super().__init__(random_state=None)
        self.init_value = init_value
        self.final_value = final_value
        self.k = k
        self.exact_bounds = exact_bounds

    def evaluate(self, progress):
        val_diff = self.final_value - self.init_value
        if self.exact_bounds:
            naive_logistic = sp.special.expit(self.k * (progress - 0.5))
            l_min = sp.special.expit(-self.k * 0.5)
            l_max = sp.special.expit(self.k * 0.5)
            return self.init_value + val_diff * (naive_logistic - l_min) / (l_max - l_min)
        else:
            return self.init_value + val_diff * sp.special.expit(self.k * (progress - 0.5))
