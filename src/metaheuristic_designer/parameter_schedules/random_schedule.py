from ..schedulable_parameter import SchedulableParameter


class RandomSchedule(SchedulableParameter):
    def __init__(self, init_value, final_value, random_state=None):
        super().__init__(random_state=random_state)
        self.init_value = init_value
        self.final_value = final_value

    def evaluate(self, _progress):
        return self.random_state.uniform(self.init_value, self.final_value)
