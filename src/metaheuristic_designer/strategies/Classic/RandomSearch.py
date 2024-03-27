from __future__ import annotations
from ...operators import OperatorReal
from ..HillClimb import HillClimb


class RandomSearch(HillClimb):
    """
    Random search
    """

    def __init__(self, initializer, name="RandomSearch"):
        random_op = OperatorReal("Random")
        super().__init__(initializer, random_op, name=name)
