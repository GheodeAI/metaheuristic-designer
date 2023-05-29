from __future__ import annotations
from .ParamScheduler import ParamScheduler
from abc import ABC, abstractmethod


class Operator(ABC):
    """
    Abstract Operator class
    """

    last_id = 0

    def __init__(self, params: Union[ParamScheduler, dict], name=None):
        """
        Constructor for the Operator class
        """

        self.id = Operator.last_id
        Operator.last_id += 1

        self.param_scheduler = None

        self.name = name

        if params is None:

            # Default parameters
            self.params = {}
        else:
            if "method" in params:
                params["method"] = params["method"].lower()

            if isinstance(params, ParamScheduler):
                self.param_scheduler = params
                self.params = self.param_scheduler.get_params()
            else:
                self.params = params

    def __call__(self, solution: Individual, population: List[Individual], objfunc: ObjectiveFunc, global_best: Individual, initializer: Initializer) -> Individual:
        """
        A shorthand for calling the 'evolve' method
        """

        return self.evolve(solution, population, objfunc, global_best, initializer)

    def step(self, progress: float):
        """
        Updates the parameters of the method using a paramater scheduler if it exists
        """

        if self.param_scheduler:
            self.param_scheduler.step(progress)
            self.params = self.param_scheduler.get_params()
    
    def get_state(self) -> dict:
        """
        Gets the current state of the algorithm as a dictionary.
        """
        
        data = {
            "name": self.name
        }

        if self.param_scheduler:
            data["param_scheduler"] = self.param_scheduler.get_state()
            data["params"] = self.param_scheduler.get_params()
            data["params"].pop("function", None)
        elif self.params:
            data["params"] = self.params
            data["params"].pop("function", None)
        
        return data

    @abstractmethod
    def evolve(self, solution: Individual, population: List[Individual], objfunc: ObjectiveFunc, global_best: Individual, initializer: Initializer) -> Individual:
        """
        Evolves a solution with a different strategy depending on the type of substrate
        """
