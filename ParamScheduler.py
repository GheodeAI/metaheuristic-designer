import random
import numpy as np
import math 


class ParamScheduler:
    """
    This class is responsible of varying the parameters of an algorithm over time.
    """

    def __init__(self, strategy: str, param_schedule: dict):
        """
        Constructor for the ParamScheduler class
        """

        self.param_schedule = param_schedule
        self.strategy = strategy

        self.reset()


    def __getitem__(self, idx: str) -> type:
        """
        Gets the current value of a parameter given it's name
        """

        return self.current_params[idx]
    

    def __setitem__(self, idx: str, value: type):
        """
        Sets the current value of a parameter given it's name
        """

        self.current_params[idx] = value
    
    def __contains__(self, value: str) -> bool:
        """
        Gets wether an element is inside the dictionary or not
        """

        return value in self.current_params
    

    def reset(self):
        """
        Sets all the parameters to their initial values
        """

        self.current_params = {}
        self.current_params.update(self.param_schedule)

        for key in self.param_schedule:
            if type(self.param_schedule[key]) in (list, tuple):
                self.current_params[key] = self.param_schedule[key][0]    


    def get_params(self) -> dict:
        """
        Returns a dictionary containing the current parameters
        """

        return self.current_params


    def step(self, progress: float):
        """
        Changes the values of the parameters interpolating between the initial and final values.
        """

        if self.strategy == "Linear":

            for key in self.param_schedule:
                if type(self.param_schedule[key]) in (list, tuple):
                    start_param = self.param_schedule[key][0]
                    end_param = self.param_schedule[key][1]
                    self.current_params[key] = (1-progress)*start_param + progress*end_param
                
        elif self.strategy == "Exp":
            # with f(x) = k·e^{a·x}+b,  f(0) = p[0],  f(1) = p[1] 
            for key in self.param_schedule:
                if type(self.param_schedule[key]) in (list, tuple):
                    start_param = self.param_schedule[key][0]
                    end_param = self.param_schedule[key][1]
                    k = 1
                    a = (end_param - start_param)/math.exp(k)
                    b = start_param
                    self.current_params[key] = a*math.exp(k*progress) + b


if __name__ == "__main__":
    a = {
        "a": "Gauss",
        "b": [0.01, 0.0000001]
    }

    p = ParamScheduler("Linear", a)
    for i in np.linspace(0,1,101):
        p.step(i)
        p["a"] = p["a"][0] + p["a"]
        print(p.get_params())
        print(p["a"])



