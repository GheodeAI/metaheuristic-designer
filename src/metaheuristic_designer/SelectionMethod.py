from __future__ import annotations
from abc import ABC, abstractmethod
from .ParamScheduler import ParamScheduler
from .Population import Population


class SelectionMethod(ABC):
    """
    Abstract Selection Method class.

    This class defines the structure for individual selection methods.

    Parameters
    ----------
    params: ParamScheduler or dict, optional
        Dictionary of parameters to define the behaviour of the selection method.
    name: str, optional
        The name that will be assigned to this selection method.
    """

    def __init__(
        self,
        params: ParamScheduler | dict = None,
        name: str = None,
    ):
        """
        Constructor for the SurvivorSelection class
        """

        self.name = name

        self.param_scheduler = None
        if params is None:
            self.params = {"amount": 10, "p": 0.1}
        elif isinstance(params, ParamScheduler):
            self.param_scheduler = params
            self.params = self.param_scheduler.get_params()
        else:
            self.params = params
        self.last_selection_idx = None

    def __call__(self, population: Population, offspring: Population = None) -> Population:
        """
        Shorthand for calling the 'select' method
        """

        return self.select(population, offspring)

    @abstractmethod
    def select(self, population: Population, offspring: Population) -> Population:
        """
        Takes a population with its offspring and returns the individuals that survive
        to produce the next generation.

        Parameters
        ----------
        population: Population
            Population of individuals that will be selected.
        offspring: Population
            Newly generated individuals to be selected.

        Returns
        -------
        selected: Population
            List of selected individuals.
        """

    def step(self, progress: float):
        """
        Updates the parameters of the method using a paramater scheduler if it exists

        Parameters
        ----------
        progress: float
            Estimated percentage of the progress of the algorithm (0 if it just started and 1 if it has ended).
        """

        if self.param_scheduler:
            self.param_scheduler.step(progress)
            self.params = self.param_scheduler.get_params()

            if "amount" in self.params:
                self.params["amount"] = round(self.params["amount"])

            if "maxPopSize" in self.params:
                self.params["maxPopSize"] = round(self.params["maxPopSize"])

    def set_param(self, **kwargs):
        """"""
        self.params.update(kwargs)

    def get_state(self):
        """
        Gets the current state of the algorithm as a dictionary.
        """

        data = {"name": self.name}

        if self.param_scheduler:
            data["param_scheduler"] = self.param_scheduler.get_state()
            data["params"] = self.param_scheduler.get_params()
        elif self.params:
            data["params"] = self.params

        return data
