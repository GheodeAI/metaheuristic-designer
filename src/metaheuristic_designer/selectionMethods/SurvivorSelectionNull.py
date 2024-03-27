from __future__ import annotations
import enum
from enum import Enum
from ..ParamScheduler import ParamScheduler
from ..SelectionMethod import SelectionMethod
from .survivor_selection_functions import *


class SurvivorSelectionNull(SelectionMethod):
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
        params: Union[ParamScheduler, dict] = None,
        padding: bool = False,
        name: str = None,
    ):
        """
        Constructor for the SurvivorSelection class
        """

        if name is None:
            name = "Nothing"

        super().__init__(params, padding, name)

    def select(self, popul: List[Individual], offspring: List[Individual]) -> List[Individual]:
        return offspring
