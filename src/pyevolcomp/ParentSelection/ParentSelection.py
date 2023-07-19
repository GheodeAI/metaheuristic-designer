from __future__ import annotations
import enum
from enum import Enum
from ..ParamScheduler import ParamScheduler
from .parent_selection_functions import *


class ParentSelMethod(Enum):
    TOURNAMENT = enum.auto()
    BEST = enum.auto()
    RANDOM = enum.auto()
    ROULETTE = enum.auto()
    SUS = enum.auto()
    NOTHING = enum.auto()

    @staticmethod
    def from_str(str_input):

        str_input = str_input.lower()

        if str_input not in parent_sel_map:
            raise ValueError(f"Survivor selection method \"{str_input}\" not defined")

        return parent_sel_map[str_input]


parent_sel_map = {
    "tournament": ParentSelMethod.TOURNAMENT,
    "best": ParentSelMethod.BEST,
    "random": ParentSelMethod.RANDOM,
    "roulette": ParentSelMethod.ROULETTE,
    "sus": ParentSelMethod.SUS,
    "nothing": ParentSelMethod.NOTHING
}


class ParentSelection:
    """
    Parent selection methods
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
        
        if self.method in [ParentSelMethod.ROULETTE, ParentSelMethod.SUS]:
            self.params["method"] = SelectionDist.from_str(self.params["method"])
            if "F" not in self.params:
                self.params["F"] = None

    def __call__(self, population: List[Individual]) -> List[Individual]:
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
        elif self.params:
            data["params"] = self.params
        
        return data           

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
        
        elif self.method == ParentSelMethod.RANDOM:
            parents, order = uniform_selection(population, self.params["amount"])
        
        elif self.method == ParentSelMethod.ROULETTE:
            parents, order = roulette(population, self.params["amount"], self.params["method"], self.params["F"])
        
        elif self.method == ParentSelMethod.SUS:
            parents, order = sus(population, self.params["amount"], self.params["method"], self.params["F"])

        elif self.method == ParentSelMethod.NOTHING:
            parents, order = population, range(len(population))

        return parents, order
