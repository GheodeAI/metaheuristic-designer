from __future__ import annotations
from ...operators import create_operator
from ..hill_climb import HillClimb


class RandomSearch(HillClimb):
    """
    Random search
    """

    def __init__(self, initializer, name="RandomSearch", **kwargs):
        super().__init__(initializer, operator=create_operator(method="random.random"), name=name, **kwargs)
