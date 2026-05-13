from ..schedulable_parameter import SchedulableParameter


class ThresholdSchedule(SchedulableParameter):
    def __init__(self, init_value, final_value, threshold=0.5):
        super().__init__(random_state=None)
        self.init_value = init_value
        self.final_value = final_value
        self.threshold = threshold

    def evaluate(self, progress):
        return self.init_value if progress < self.threshold else self.final_value
