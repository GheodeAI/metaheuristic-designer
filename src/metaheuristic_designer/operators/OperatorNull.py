from __future__ import annotations
import numpy as np
from ..Operator import Operator
from .vector_operator_functions import *
from ..ParamScheduler import ParamScheduler
from copy import copy
import enum
from enum import Enum
from ..utils import RAND_GEN


class OperatorNull(Operator):
    """
    Operator class that applies a custom operator specified as a function.

    Parameters
    ----------
    fn: callable
        Function that will be applied when operating on an individual.
    params: ParamScheduler or dict, optional
        Dictionary of parameters to define the operator.
    name: str, optional
        Name that is associated with the operator.
    """

    def __init__(self, name: str = None):
        """
        Constructor for the OperatorReal class
        """

        if name is None:
            name = "Nothing"

        super().__init__({}, name)

    def evolve(self, indiv, *args):
        return indiv
