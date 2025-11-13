from __future__ import annotations
from ..initializer import Initializer
from ..search_strategy import SearchStrategy


class NoSearch(SearchStrategy):
    """
    Debug Algorithm that does nothing
    """

    def __init__(self, initializer: Initializer, name: str = "No search"):
        super().__init__(initializer, params={}, name=name)

    def perturb(self, parents, **kwargs):
        return parents
