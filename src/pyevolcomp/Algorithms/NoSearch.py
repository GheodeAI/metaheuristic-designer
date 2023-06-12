from __future__ import annotations
from ..Algorithm import Algorithm


class NoSearch(Algorithm):
    """
    Debug Algorithm that does nothing
    """

    def __init__(self, pop_init: Initializer, name: str = "No search"):
        super().__init__(pop_init, params={}, name=name)

    def perturb(self, parent_list, pop_init, objfunc, progress=0, history=None):
        return parent_list


