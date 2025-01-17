from __future__ import annotations
from copy import copy
from ..Operator import Operator


class OperatorNull(Operator):
    """
    Operator class that returns the individual without changes.
    Surprisingly useful.

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
        Constructor for the OperatorNull class
        """

        if name is None:
            name = "Nothing"

        super().__init__({}, name)

    def evolve(self, population, *args):
        return copy(population)
