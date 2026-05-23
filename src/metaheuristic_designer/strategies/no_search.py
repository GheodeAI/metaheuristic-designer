"""
No-op strategy that returns the population unchanged (debug / baseline).
"""

from __future__ import annotations
from copy import copy
from ..initializer import Initializer
from ..search_strategy import SearchStrategy
from ..population import Population


class NoSearch(SearchStrategy):
    """
    Debug strategy that does nothing.

    The population is never modified.  Useful as a baseline or for
    testing other components in isolation.

    Parameters
    ----------
    initializer : Initializer
        Population initializer.
    name : str, optional
        Display name (default ``"No search"``).
    **kwargs
        Forwarded to :class:`SearchStrategy`.
    """

    def __init__(self, initializer: Initializer, name: str = "No search", **kwargs):
        super().__init__(initializer, name=name, **kwargs)

    def iterate(self, population: Population) -> Population:
        return copy(population)
