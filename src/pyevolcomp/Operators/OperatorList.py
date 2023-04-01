from ..Operator import Operator
from .list_operator_functions import *
from .vector_operator_functions import *
from ..ParamScheduler import ParamScheduler
from typing import Union
from copy import copy
from .OperatorReal import OperatorReal, _real_ops

_list_ops = [
    "expand",
    "shrink",
    "nothing"
]

_list_ops = _list_ops

class OperatorList(Operator):
    """
    Operator class that has continuous mutation and cross methods
    """

    def __init__(self, method: str, params: Union[ParamScheduler, dict]=None, name=None):
        """
        Constructor for the OperatorReal class
        """

        super().__init__(method, params, name)

        if self.method not in _list_ops and not self.is_vec_op:
            raise ValueError(f"Real operator \"{self.method}\" not defined")
    
    
    def evolve(self, indiv, population, objfunc, global_best):
        """
        Evolves a solution with a different strategy depending on the type of operator
        """

        new_indiv = copy(indiv)
        others = [i for i in population if i != indiv]
        if len(others) > 1:
            indiv2 = random.choice(others)
        else:
            indiv2 = indiv
        
        params = copy(self.params)

        if self.method == "expand":
            nex_indiv.genotype = expand(new_indiv.genotype, params["N"], params["method"])
        elif self.method == "shrink":
            nex_indiv.genotype = shrink(new_indiv.genotype, params["N"], params["method"])
        
        return new_indiv

