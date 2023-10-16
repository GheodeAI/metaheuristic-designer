from __future__ import annotations
from ..Operator import Operator
from .OperatorReal import OperatorReal, real_ops_map
from .list_operator_functions import *
from .vector_operator_functions import *
from copy import copy
import enum
from enum import Enum


class ListOpMethods(Enum):
    EXPAND = enum.auto()
    SHRINK = enum.auto()
    NOTHING = enum.auto()

    @staticmethod
    def from_str(str_input):
        str_input = str_input.lower()

        if str_input not in list_ops_map:
            raise ValueError(f'List operator "{str_input}" not defined')

        return list_ops_map[str_input]


list_ops_map = {
    "expand": ListOpMethods.EXPAND,
    "shrink": ListOpMethods.SHRINK,
    "nothing": ListOpMethods.NOTHING,
}


class OperatorList(Operator):
    """
    Operator class that works on variable length lists.

    Parameters
    ----------
    method: str
        Type of operator that will be applied.
    params: ParamScheduler or dict, optional
        Dictionary of parameters to define the operator.
    name: str, optional
        Name that is associated with the operator.
    """

    def __init__(
        self, method: str, params: Union[ParamScheduler, dict] = None, name: str = None
    ):
        """
        Constructor for the OperatorReal class
        """

        if name is None:
            name = method

        super().__init__(params, name)

        self.method = ListOpMethods.from_str(method)

    def evolve(self, indiv, population, objfunc, global_best, initializer):
        new_indiv = copy(indiv)

        params = copy(self.params)

        if self.method == ListOpMethods.EXPAND:
            nex_indiv.genotype = expand(
                new_indiv.genotype,
                params["N"],
                params["method"],
                params["maxlen"],
                params["generator"],
            )
        elif self.method == ListOpMethods.SHRINK:
            nex_indiv.genotype = shrink(
                new_indiv.genotype, params["N"], params["method"]
            )

        return new_indiv
