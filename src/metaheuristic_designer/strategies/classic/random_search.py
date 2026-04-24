from __future__ import annotations
from ...operators import create_operator
from ..hill_climb import HillClimb


class RandomSearch(HillClimb):
    """
    Random search
    """

    def __init__(self, initializer, name="RandomSearch"):
        random_op = create_operator(method="Random.random")
        super().__init__(initializer, operator=random_op, name=name)
