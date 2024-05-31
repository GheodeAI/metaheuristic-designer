from __future__ import annotations
import enum
from enum import Enum
from ..ParamScheduler import ParamScheduler
from ..SelectionMethod import SelectionMethod
from .survivor_selection_functions import *


class SurvSelMultiMethod(Enum):
    GENERATIONAL = enum.auto()
    NONDOMSORT = enum.auto()

    @staticmethod
    def from_str(str_input):
        str_input = str_input.lower()

        if str_input not in surv_method_map:
            raise ValueError(f'Parent selection method "{str_input}" not defined')

        return surv_method_map[str_input]


surv_method_map = {
    "generational": SurvSelMultiMethod.GENERATIONAL,
    "nothing": SurvSelMultiMethod.GENERATIONAL,
    "nondom": SurvSelMultiMethod.NONDOMSORT,
    "non-dominated-sorting": SurvSelMultiMethod.NONDOMSORT,
}


class SurvivorSelectionMulti(SelectionMethod):
    """
    Survivor selection methods.

    Selects the individuals that will remain for the next generation of our algorithm.

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
        params: Union[ParamScheduler, dict] = None,
        padding: bool = False,
        name: str = None,
    ):
        """
        Constructor for the SurvivorSelection class
        """

        if name is None:
            name = method

        super().__init__(params, padding, name)

        self.method = SurvSelMethod.from_str(method)

    def select(self, popul: List[Individual], offspring: List[Individual]) -> List[Individual]:
        result = []
        if self.method == SurvSelMethod.GENERATIONAL:
            result = offspring

        if self.method == SurvSelMethod.NONDOMSORT:
            result = non_dominated_sorting(popul, offspring)
        
        return result

