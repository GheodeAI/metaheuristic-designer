from __future__ import annotations
from abc import ABC, abstractmethod


class Operator(ABC):
    """
    Abstract Operator class
    """

    def __init__(self, params: Union[ParamScheduler, dict], name=None):
        """
        Constructor for the Operator class
        """

        self.param_scheduler = None

        self.name = name

        if params is None:

            # Default parameters
            self.params = {
                "F": 0.5,
                "Cr": 0.8,
                "N": 5,
                "Nindiv": 5,
                "P": 0.1,
                "method": "gauss",
                "temp_ch": 10,
                "iter": 20,
                "Low": -10,
                "Up": 10,
                "epsilon": 0.1,
                "tau": 0.1,
                "tau_multiple": 0.1,
                "a": 0.1,
                "b": 0.1,
                "d": 0.1,
                "g": 0.1,
                "w": 0.7,
                "c1": 1.5,
                "c2": 1.5,
                "function": lambda x, y, z, w: x.genotype
            }
        else:
            if "method" in params:
                params["method"] = params["method"].lower()

            if isinstance(params, ParamScheduler):
                self.param_scheduler = params
                self.params = self.param_scheduler.get_params()
            else:
                self.params = params

    def __call__(self, solution: Individual, population: List[Individual], objfunc: ObjectiveFunc, global_best: Individual) -> Individual:
        """
        A shorthand for calling the 'evolve' method
        """

        return self.evolve(solution, population, objfunc, global_best)

    def step(self, progress: float):
        """
        Updates the parameters of the method using a paramater scheduler if it exists
        """

        if self.param_scheduler:
            self.param_scheduler.step(progress)
            self.params = self.param_scheduler.get_params()

    @abstractmethod
    def evolve(self, solution: Individual, population: List[Individual], objfunc: ObjectiveFunc, global_best: Individual) -> Individual:
        """
        Evolves a solution with a different strategy depending on the type of substrate
        """
