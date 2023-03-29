from ..Operator import Operator
from .list_operator_functions import *
from ..ParamScheduler import ParamScheduler
from typing import Union
from copy import copy
from .OperatorReal import OperatorReal, _real_ops

_list_ops = [
    "expand",
    "shrink"
]

_list_ops = _list_ops + _real_ops

class OperatorList(Operator):
    """
    Operator class that has continuous mutation and cross methods
    """

    def __init__(self, name: str, params: Union[ParamScheduler, dict]=None):
        """
        Constructor for the OperatorReal class
        """

        if name.lower() not in _list_ops:
            raise ValueError(f"Real operator \"{self.name}\" not defined")

        super().__init__(name, params)
    
    
    def evolve(self, indiv, population, objfunc, global_best=None):
        """
        Evolves a solution with a different strategy depending on the type of operator
        """