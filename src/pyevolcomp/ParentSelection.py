from __future__ import annotations
from enum import Enum
from .ParamScheduler import ParamScheduler
from .parent_selection_functions import *


class ParentSelMethod(Enum):
    TOURNAMENT = 1
    BEST = 2
    NOTHING = 3

    @staticmethod
    def from_str(str_input):

        str_input = str_input.lower()

        if str_input not in parent_sel_map:
            raise ValueError(f"Survivor selection method \"{str_input}\" not defined")

        return parent_sel_map[str_input]


parent_sel_map = {
    "tournament": ParentSelMethod.TOURNAMENT,
    "best": ParentSelMethod.BEST,
    "nothing": ParentSelMethod.NOTHING
}


class ParentSelection:
    """
    Operator class that has continuous mutation and cross methods
    """

    def __init__(self, method: str, params: Union[ParamScheduler, dict] = None, name: str = None):
        """
        Constructor for the ParentSelection class
        """

        if name is None:
            self.name = method

        self.method = ParentSelMethod.from_str(method)

        self.param_scheduler = None
        if params is None:
            self.params = {"amount": 10, "p": 0.1}
        elif isinstance(params, ParamScheduler):
            self.param_scheduler = params
            self.params = self.param_scheduler.get_params()
        else:
            self.params = params

    def __call__(self, population: List[Individual]):
        """
        Shorthand for calling the 'select' method
        """

        return self.select(population)

    def step(self, progress: float):
        """
        Updates the parameters of the method using a paramater scheduler if it exists
        """

        if self.param_scheduler:
            self.param_scheduler.step(progress)
            self.params = self.param_scheduler.get_params()

            if "amount" in self.params:
                self.params["amount"] = round(self.params["amount"])

    def select(self, population: List[Individual]) -> List[Individual]:
        """
        Selects a subsection of the population along with the indices of each individual in the original population
        """

        parents = []
        order = []
        if self.method == ParentSelMethod.TOURNAMENT:
            parents, order = prob_tournament(population, self.params["amount"], self.params["p"])

        elif self.method == ParentSelMethod.BEST:
            parents, order = select_best(population, self.params["amount"])

        elif self.method == ParentSelMethod.NOTHING:
            parents, order = population, range(len(population))

        return parents, order
