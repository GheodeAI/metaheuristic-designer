from __future__ import annotations
import enum
from enum import Enum
from ..ParamScheduler import ParamScheduler
from ..SelectionMethod import SelectionMethod
from .survivor_selection_functions import *


class SurvSelMethod(Enum):
    ELITISM = enum.auto()
    COND_ELITISM = enum.auto()
    GENERATIONAL = enum.auto()
    ONE_TO_ONE = enum.auto()
    PROB_ONE_TO_ONE = enum.auto()
    MU_PLUS_LAMBDA = enum.auto()
    MU_COMMA_LAMBDA = enum.auto()
    CRO = enum.auto()

    @staticmethod
    def from_str(str_input):
        str_input = str_input.lower()

        if str_input not in surv_method_map:
            raise ValueError(f'Parent selection method "{str_input}" not defined')

        return surv_method_map[str_input]


surv_method_map = {
    "elitism": SurvSelMethod.ELITISM,
    "condelitism": SurvSelMethod.COND_ELITISM,
    "generational": SurvSelMethod.GENERATIONAL,
    "nothing": SurvSelMethod.GENERATIONAL,
    "one-to-one": SurvSelMethod.ONE_TO_ONE,
    "hillclimb": SurvSelMethod.ONE_TO_ONE,
    "prob-one-to-one": SurvSelMethod.PROB_ONE_TO_ONE,
    "probhillclimb": SurvSelMethod.PROB_ONE_TO_ONE,
    "(m+n)": SurvSelMethod.MU_PLUS_LAMBDA,
    "keepbest": SurvSelMethod.MU_PLUS_LAMBDA,
    "(m,n)": SurvSelMethod.MU_COMMA_LAMBDA,
    "keepoffsping": SurvSelMethod.MU_COMMA_LAMBDA,
    "cro": SurvSelMethod.CRO,
}


class SurvivorSelection(SelectionMethod):
    """
    Survivor selection methods

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

    def select(
        self, popul: List[Individual], offspring: List[Individual]
    ) -> List[Individual]:
        result = []
        if self.method == SurvSelMethod.ELITISM:
            result = elitism(popul, offspring, self.params["amount"])

        elif self.method == SurvSelMethod.COND_ELITISM:
            result = cond_elitism(popul, offspring, self.params["amount"])

        elif self.method == SurvSelMethod.GENERATIONAL:
            result = offspring

        elif self.method == SurvSelMethod.ONE_TO_ONE:
            result = one_to_one(popul, offspring)

        elif self.method == SurvSelMethod.PROB_ONE_TO_ONE:
            result = prob_one_to_one(popul, offspring, self.params["p"])

        elif self.method == SurvSelMethod.MU_PLUS_LAMBDA:
            result = lamb_plus_mu(popul, offspring)

        elif self.method == SurvSelMethod.MU_COMMA_LAMBDA:
            result = lamb_comma_mu(popul, offspring)

        elif self.method == SurvSelMethod.CRO:
            result = cro_selection(
                popul,
                offspring,
                self.params["Fd"],
                self.params["Pd"],
                self.params["attempts"],
                self.params["maxPopSize"],
            )

        return result
