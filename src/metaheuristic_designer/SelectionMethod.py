from __future__ import annotations
from abc import ABC, abstractmethod
from .ParamScheduler import ParamScheduler


class SelectionMethod(ABC):
    """
    Survivor selection methods

    Parameters
    ----------
    params: ParamScheduler or dict, optional
        Dictionary of parameters to define the behaviour of the selection method.
    padding: bool, optional
        Whether to fill the entire list of selected individuals to match the size of the original one.
    name: str, optional
        The name that will be assigned to this selection method.
    """

    def __init__(
        self,
        params: Union[ParamScheduler, dict] = None,
        padding: bool = False,
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

    def __call__(
        self, popul: List[Individual], offspring: List[Individual] = None
    ) -> List[Individual]:
        """
        Shorthand for calling the 'select' method
        """

        return self.select(popul, offspring)

    @abstractmethod
    def select(
        self, population: List[Individual], offspring: List[Individual] = None
    ) -> List[Individual]:
        """
        Takes a population with its offspring and returns the individuals that survive
        to produce the next generation.

        Parameters
        ----------
        population: List[Individual]
            Population of individuals that will be selected.
        offspring: List[Individual]
            Newly generated individuals to be selected.

        Returns
        -------
        selected: List[Individual]
            List of selected individuals.
        """

    def step(self, progress: float):
        """
        Updates the parameters of the method using a paramater scheduler if it exists
        """

        if self.param_scheduler:
            self.param_scheduler.step(progress)
            self.params = self.param_scheduler.get_params()

            if "amount" in self.params:
                self.params["amount"] = round(self.params["amount"])

            if "maxPopSize" in self.params:
                self.params["maxPopSize"] = round(self.params["maxPopSize"])

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
