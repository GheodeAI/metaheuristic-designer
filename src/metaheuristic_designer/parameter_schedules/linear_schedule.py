from ..schedulable_parameter import SchedulableParameter


class LinearSchedule(SchedulableParameter):
    def __init__(self, init_value, final_value):
        super().__init__(random_state=None)
        self.init_value = init_value
        self.final_value = final_value

    def evaluate(self, progress):
        return (1 - progress) * self.init_value + progress * self.final_value
