from __future__ import annotations
from ..SearchStrategy import SearchStrategy


class NoSearch(SearchStrategy):
    """
    Debug Algorithm that does nothing
    """

    def __init__(self, pop_init: Initializer, name: str = "No search"):
        super().__init__(pop_init, params={}, name=name)

    def perturb(self, parent_list, objfunc, **kwargs):
        return parent_list
