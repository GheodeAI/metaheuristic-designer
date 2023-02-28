from abc import ABC, abstractmethod

from ..ParamScheduler import *


class Operator(ABC):
    """
    Abstract Operator class
    """

    def __init__(self, name, params):
        """
        Constructor for the Operator class
        """

        self.name = name
        self.param_scheduler = None

        if params is None:

            # Default parameters
            self.params = {
                "F": 0.5, 
                "Cr": 0.8,
                "Par":0.1,
                "N":5,
                "method": "Gauss",
                "temp_ch":10,
                "iter":20,
                "Low":-10,
                "Up":10
            }
        elif isinstance(params, ParamScheduler):
            self.param_scheduler = params
            self.params = self.param_scheduler.get_params()
        else:
            self.params = params
    

    def __call__(self, solution, population, objfunc):
        """
        A shorthand for calling the 'evolve' method
        """

        return self.evolve(solution, population, objfunc)
    
    
    def step(self, progress):
        """
        Updates the parameters of the method using a paramater scheduler if it exists
        """

        if self.param_scheduler:
            self.param_scheduler.step(progress)
            self.params = self.param_scheduler.get_params()


    @abstractmethod
    def evolve(self, solution, population, objfunc):
        """
        Evolves a solution with a different strategy depending on the type of substrate
        """
