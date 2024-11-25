from __future__ import annotations
from ..Population import Population
from ..ParamScheduler import ParamScheduler
from ..SelectionMethod import SelectionMethod


class ParentSelectionNull(SelectionMethod):
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
        params: ParamScheduler | dict = None,
        padding: bool = False,
        name: str = None,
    ):
        """
        Constructor for the ParentSelection class
        """

        if name is None:
            name = "Nothing"

        super().__init__(params, padding, name)

    def select(self, population: Population, offspring: Population = None) -> Population:
        return population
