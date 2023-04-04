from __future__ import annotations
from ..Operator import Operator
from .OperatorReal import OperatorReal, _real_ops
from .list_operator_functions import *
from .vector_operator_functions import *
from copy import copy
from enum import Enum


class ListOpMethods(Enum):
    EXPAND = 1
    SHRINK = 2
    NOTHING = 3

    @staticmethod
    def from_str(str_input):

        str_input = str_input.lower()

        if str_input not in list_ops_map:
            raise ValueError(f"List operator \"{str_input}\" not defined")

        return list_ops_map[str_input]


list_ops_map = {
    "expand": ListOpMethods.EXPAND,
    "shrink": ListOpMethods.SHRINK,
    "nothing": ListOpMethods.NOTHING,
}


class OperatorList(Operator):
    """
    Operator class that has continuous mutation and cross methods
    """

    def __init__(self, method: str, params: Union[ParamScheduler, dict] = None, name: str = None):
        """
        Constructor for the OperatorReal class
        """

        if name is None:
            name = method

        super().__init__(params, name)

        self.method = ListOpMethods.from_str(method)

    def evolve(self, indiv, population, objfunc, global_best):
        """
        Evolves a solution with a different strategy depending on the type of operator
        """

        new_indiv = copy(indiv)
        # others = [i for i in population if i != indiv]
        # if len(others) > 1:
        #     indiv2 = random.choice(others)
        # else:
        #     indiv2 = indiv

        params = copy(self.params)

        if self.method == ListOpMethods.EXPAND:
            nex_indiv.genotype = expand(new_indiv.genotype, params["N"], params["method"], params["maxlen"])
        elif self.method == ListOpMethods.SHRINK:
            nex_indiv.genotype = shrink(new_indiv.genotype, params["N"], params["method"])

        return new_indiv
