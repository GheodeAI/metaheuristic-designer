from __future__ import annotations
from enum import Enum
from .ParamScheduler import ParamScheduler
from .survivor_selection_functions import *


class SurvSelMethod(Enum):
    ELITISM = 1
    COND_ELITISM = 2
    GENERATIONAL = 3
    ONE_TO_ONE = 4
    MU_PLUS_LAMBDA = 5
    MU_COMMA_LAMBDA = 6
    CRO = 7

    @staticmethod
    def from_str(str_input):

        str_input = str_input.lower()

        if str_input not in surv_method_map:
            raise ValueError(f"Parent selection method \"{str_input}\" not defined")

        return surv_method_map[str_input]


surv_method_map = {
    "elitism": SurvSelMethod.ELITISM,
    "condelitism": SurvSelMethod.COND_ELITISM,
    "generational": SurvSelMethod.GENERATIONAL,
    "nothing": SurvSelMethod.GENERATIONAL,
    "one-to-one": SurvSelMethod.ONE_TO_ONE,
    "(m+n)": SurvSelMethod.MU_PLUS_LAMBDA,
    "(m,n)": SurvSelMethod.MU_COMMA_LAMBDA,
    "cro": SurvSelMethod.CRO
}


class SurvivorSelection:
    """
    Operator class that has continuous mutation and cross methods
    """

    def __init__(self, method: str, params: Union[ParamScheduler, dict] = None, name: str = None, padding: bool = False):
        """
        Constructor for the SurvivorSelection class
        """

        if name is None:
            self.name = method

        self.method = SurvSelMethod.from_str(method)

        self.param_scheduler = None
        if params is None:
            self.params = {"amount": 10, "p": 0.1}
        elif isinstance(params, ParamScheduler):
            self.param_scheduler = params
            self.params = self.param_scheduler.get_params()
        else:
            self.params = params

    def __call__(self, popul: List[Individual], offspring: List[Individual]) -> List[Individual]:
        """
        Shorthand for calling the 'select' method
        """

        return self.select(popul, offspring)

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

        data = {
            "name": self.name
        }

        if self.param_scheduler:
            data["param_scheduler"] = self.param_scheduler.get_state()
            data["params"] = self.param_scheduler.get_params()
        else:
            data["params"] = self.params
        
        return data
    
    
            

    def select(self, popul: List[Individual], offspring: List[Individual]) -> List[Individual]:     
        """
        Takes a population with its offspring and returns the individuals that survive
        to produce the next generation.
        """

        result = []
        if self.method == SurvSelMethod.ELITISM:
            result = elitism(popul, offspring, self.params["amount"])

        elif self.method == SurvSelMethod.COND_ELITISM:
            result = cond_elitism(popul, offspring, self.params["amount"])

        elif self.method == SurvSelMethod.GENERATIONAL:
            result = offspring

        elif self.method == SurvSelMethod.ONE_TO_ONE:
            result = one_to_one(popul, offspring)

        elif self.method == SurvSelMethod.MU_PLUS_LAMBDA:
            result = lamb_plus_mu(popul, offspring)

        elif self.method == SurvSelMethod.MU_COMMA_LAMBDA:
            result = lamb_comma_mu(popul, offspring)
        
        elif self.method == SurvSelMethod.CRO:
            result = cro_selection(popul, offspring, self.params["Fd"], self.params["Pd"], self.params["attempts"], self.params["maxPopSize"])

        return result
