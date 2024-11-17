from __future__ import annotations
from .ParamScheduler import ParamScheduler
from .encodings import DefaultEncoding
from abc import ABC, abstractmethod


class Operator(ABC):
    """
    Abstract Operator class.

    This class modifies the genotype of one individual in order to perform some optimization task.

    Parameters
    ----------
    params: ParamScheduler or dict, optional
        Dictionary of parameters to define the operator.
    name: str, optional
        Name that is associated with the operator.
    encoding: Encoding, optional
        Postprocessing to the operator output.
    """

    _last_id = 0

    def __init__(self, params: ParamScheduler | dict = None, name: str = None, encoding: Encoding = None):
        """
        Constructor for the Operator class.
        """

        self.id = Operator._last_id
        Operator._last_id += 1

        self.param_scheduler = None

        self.name = name

        if encoding is None:
            encoding = DefaultEncoding()
        self.encoding = encoding

        if params is None:
            self.params = {}
        elif params == "default":
            self.params = {
                "F": 0.5,
                "Cr": 0.8,
                "N": 5,
                "Nindiv": 5,
                "P": 0.1,
                "distrib": "gauss",
                "temp_ch": 10,
                "iter": 20,
                "Low": -10,
                "Up": 10,
                "p": 0.5,
                "mu": 2,
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
                "function": lambda x, y, z: x,
            }
        else:
            if "method" in params:
                params["method"] = params["method"].lower()

            if isinstance(params, ParamScheduler):
                self.param_scheduler = params
                self.params = self.param_scheduler.get_params()
            else:
                self.params = params

    def __call__(
        self,
        population: Population,
        objfunc: ObjectiveFunc,
        global_best: Any,
        initializer: Initializer,
    ) -> Population:
        """
        A shorthand for calling the 'evolve' method.
        """

        return self.evolve(population, objfunc, global_best, initializer)

    def step(self, progress: float):
        """
        Updates the parameters of the method using a paramater scheduler if it exists.

        Parameters
        ----------
        progress: float
            Indicator of how close it the algorithm to finishing, 1 means the algorithm should be stopped.
        """

        if self.param_scheduler:
            self.param_scheduler.step(progress)
            self.params = self.param_scheduler.get_params()

    def get_state(self) -> dict:
        """
        Gets the current state of the algorithm as a dictionary.

        Returns
        -------
        state: dict
            The complete state of the operator.
        """

        data = {"name": self.name}

        if self.param_scheduler:
            data["param_scheduler"] = self.param_scheduler.get_state()
            data["params"] = self.param_scheduler.get_params()
            data["params"].pop("function", None)
        elif self.params:
            data["params"] = self.params
            data["params"].pop("function", None)

        return data

    @abstractmethod
    def evolve(
        self,
        population: Population,
        initializer: Initializer = None,
        global_best: Any = None,
    ) -> Population:
        """
        Evolves an population using a given strategy.

        Parameters
        ----------
        population: List[Individual]
            The population that will be used in crossing operations.
        objfunc: ObjectiveFunc
            The objective function being optimized.
        initializer: Initialize, params["longitude_max"]r
            The population initializer of the algorithm (used for randomly generating individuals).
        global_best: Individual
            The best individual found during the optimization of the algorithm

        Returns
        -------
        new_population: List[Individual]
            The modified population.
        """
    
    def evolve_single(
        self,
        indiv: Any,
        population: Population,
        initializer: Initializer = None,
        global_best: Any = None,
    ) -> Any:
        """
        Evolves an individual using a given strategy.

        Parameters
        ----------
        indiv: Individual
            Individual to be operated on.
        population: List[Individual]
            The population that will be used in crossing operations.
        objfunc: ObjectiveFunc
            The objective function being optimized.
        initializer: Initializer
            The population initializer of the algorithm (used for randomly generating individuals).
        global_best: Individual
            The best individual found during the optimization of the algorithm

        Returns
        -------
        new_individual: Individual
            The modified individual.
        """
