from __future__ import annotations
from ..initializer import Initializer
from ..search_strategy import SearchStrategy
from ..population import Population


class NoSearch(SearchStrategy):
    """
    Debug Algorithm that does nothing
    """

    def __init__(self, initializer: Initializer, name: str = "No search", **kwargs):
        super().__init__(initializer, params={}, name=name, **kwargs)

    def perturb(self, parents: Population, **kwargs) -> Population:
        return parents
