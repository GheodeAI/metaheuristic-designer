from __future__ import annotations
from ..SearchStrategy import SearchStrategy


class NoSearch(SearchStrategy):
    """
    Debug Algorithm that does nothing
    """

    def __init__(self, initializer: Initializer, name: str = "No search"):
        super().__init__(initializer, params={}, name=name)

    def perturb(self, parent_list, objfunc, **kwargs):
        return parent_list
