from __future__ import annotations
from typing import Union, List
from ...operators import OperatorReal, OperatorMeta
from ..HillClimb import HillClimb


class RandomSearch(HillClimb):
    """
    Random search
    """

    def __init__(self, pop_init, name="RandomSearch"):
        random_op = OperatorReal("Random")
        super().__init__(pop_init, random_op, name=name)
