from __future__ import annotations
from typing import Union, List
from ..Operators import OperatorReal
from .HillClimb import HillClimb


class NoSearch(HillClimb):
    """
    Debug Algorithm that does nothing
    """

    def __init__(self, pop_init: Initializer, name: str = "No search"):
        noop = OperatorReal("Nothing", {})
        super().__init__(
            pop_init, 
            noop, 
            name=name
        )

    def perturb(self, parent_list, pop_init, objfunc, progress=0, history=None):
        return parent_list
    
    def select_individuals(self, population, offspring, progress=0, history=None):
        return population


