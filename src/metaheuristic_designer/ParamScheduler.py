from __future__ import annotations
import random
import numpy as np
import math


_par_sch_methods = ["linear", "exp"]


class ParamScheduler:
    """
    This class is responsible of varying the parameters of an algorithm over time.

    Parameters
    ----------
    strategy: str
        Strategy
    param_schedule: dict
        Definition of the parameters to be varied.
    """

    def __init__(self, strategy: str, param_schedule: dict):
        """
        Constructor for the ParamScheduler class
        """

        self.strategy = strategy.lower()

        if strategy.lower() not in _par_sch_methods:
            raise ValueError(f'Parameter scheduler strategy "{self.name}" not defined')

        self.param_schedule = param_schedule

        self.reset()

    def __getitem__(self, idx: str) -> Any:
        """
        Gets the current value of a parameter given its name
        """

        return self.current_params[idx]

    def __setitem__(self, idx: str, value: Any):
        """
        Sets the current value of a parameter given its name
        """

        self.current_params[idx] = value

    def __contains__(self, value: str) -> bool:
        """
        Gets wether an element is inside the dictionary or not
        """

        return value in self.current_params

    def get(key: str, def_value: Any = None) -> Any:
        """
        Gets the current value of a parameter given its name using a default value if it's missing
        """

        return self.current_params.get(key, def_value)

    def reset(self):
        """
        Sets all the parameters to their initial values.
        """

        self.current_params = {}
        self.current_params.update(self.param_schedule)

        for key in self.param_schedule:
            if type(self.param_schedule[key]) in (list, tuple):
                self.current_params[key] = self.param_schedule[key][0]

    def get_params(self) -> dict:
        """
        Returns a dictionary containing the current parameters.

        Returns
        -------
        current_params: dict
        """

        return self.current_params

    def get_state(self) -> dict:
        """
        Gets the current state of the algorithm as a dictionary.

        Returns
        -------
        state: dict
        """

        data = {"strategy": self.strategy, "param_schedule": self.param_schedule}

        return data

    def step(self, progress: float):
        """
        Changes the values of the parameters interpolating between the initial and final values.

        Parameters
        ----------
        progress: float
        """

        if self.strategy == "linear":
            for key in self.param_schedule:
                if type(self.param_schedule[key]) in (list, tuple):
                    start_param = self.param_schedule[key][0]
                    end_param = self.param_schedule[key][1]
                    self.current_params[key] = (
                        1 - progress
                    ) * start_param + progress * end_param

        elif self.strategy == "exp":
            # with f(x) = k·e^{a·x}+b,  f(0) = p[0],  f(1) = p[1]
            for key in self.param_schedule:
                if type(self.param_schedule[key]) in (list, tuple):
                    start_param = self.param_schedule[key][0]
                    end_param = self.param_schedule[key][1]
                    k = 1
                    a = (end_param - start_param) / math.exp(k)
                    b = start_param
                    self.current_params[key] = a * math.exp(k * progress) + b
