from __future__ import annotations
from ...operators import VectorOperator
from ..HillClimb import HillClimb


class RandomSearch(HillClimb):
    """
    Random search
    """

    def __init__(self, initializer, name="RandomSearch"):
        random_op = VectorOperator("Random")
        super().__init__(initializer, operator=random_op, name=name)
