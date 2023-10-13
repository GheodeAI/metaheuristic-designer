from __future__ import annotations
import numpy as np
from ..Operator import Operator
from .vector_operator_functions import *
from ..ParamScheduler import ParamScheduler
from copy import copy
import enum
from enum import Enum
from ..utils import RAND_GEN


class OperatorLambda(Operator):
    """
    Operator class that has mutation and cross methods for real coded vectors

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
        self, fn: callable, params: Union[ParamScheduler, dict] = None, name: str = None
    ):
        """
        Constructor for the OperatorReal class
        """

        self.fn = fn

        if name is None:
            name = fn.__name__

        super().__init__(params, name)

    def evolve(self, indiv, population, objfunc, global_best, initializer):
        new_indiv = copy(indiv)
        others = [i for i in population if i != indiv]

        new_indiv.genotype = self.fn(indiv, others, objfunc, **self.params)

        return new_indiv
