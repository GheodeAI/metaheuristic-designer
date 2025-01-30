from __future__ import annotations
from copy import copy
import enum
from enum import Enum
from ..ParamScheduler import ParamScheduler
from ..SelectionMethod import SelectionMethod
from ..Population import Population
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
            raise ValueError(f'Survivor selection method "{str_input}" not defined')

        return parent_sel_map[str_input]


parent_sel_map = {
    "tournament": ParentSelMethod.TOURNAMENT,
    "best": ParentSelMethod.BEST,
    "random": ParentSelMethod.RANDOM,
    "roulette": ParentSelMethod.ROULETTE,
    "sus": ParentSelMethod.SUS,
    "nothing": ParentSelMethod.NOTHING,
}


class ParentSelection(SelectionMethod):
    """
    Parent selection methods.

    Selects the individuals that will be perturbed in this generation.

    Parameters
    ----------
    method: str
        Strategy used in the selection process.
    params: ParamScheduler or dict, optional
        Dictionary of parameters to define the behaviour of the selection method.
    padding: bool, optional
        Whether to fill the entire list of selected individuals to match the size of the original one.
    name: str, optional
        The name that will be assigned to this selection method.
    """

    def __init__(
        self,
        method: str,
        params: ParamScheduler | dict = None,
        name: str = None,
    ):
        """
        Constructor for the ParentSelection class
        """

        if name is None:
            name = method

        super().__init__(params, name)

        self.method = ParentSelMethod.from_str(method)

        if self.method in [ParentSelMethod.ROULETTE, ParentSelMethod.SUS]:
            self.params["method"] = SelectionDist.from_str(self.params.get("method", "FitnessProp"))

    def select(self, population: Population, offspring: Population = None) -> Population:
        new_population = None
        parent_idx = None
        fitness_array = population.fitness

        match self.method:
            case ParentSelMethod.TOURNAMENT:
                parent_idx = prob_tournament(fitness_array, self.params["amount"], self.params["p"])

            case ParentSelMethod.BEST:
                parent_idx = select_best(fitness_array, self.params["amount"])

            case ParentSelMethod.RANDOM:
                parent_idx = uniform_selection(fitness_array, self.params["amount"])

            case ParentSelMethod.ROULETTE:
                parent_idx = roulette(
                    fitness_array,
                    self.params["amount"],
                    self.params["method"],
                    self.params.get("F"),
                )

            case ParentSelMethod.SUS:
                parent_idx = sus(
                    fitness_array,
                    self.params["amount"],
                    self.params["method"],
                    self.params.get("F"),
                )

            case ParentSelMethod.NOTHING:
                self.last_selection_idx = range(len(population))
                new_population = population

        if new_population is None:
            self.last_selection_idx = parent_idx
            new_population = population.take_selection(parent_idx)

        return new_population
