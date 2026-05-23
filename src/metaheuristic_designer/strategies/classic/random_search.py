"""
Random search strategy (baseline).
"""

from __future__ import annotations

from ...initializer import Initializer
from ...operators import create_operator
from ..static_population_strategy import StaticPopulationStrategy


class RandomSearch(StaticPopulationStrategy):
    """
    Random search algorithm.

    Each iteration replaces the current population with completely
    new random individuals (via a ``random.random`` operator).  No
    perturbation of existing solutions occurs.

    Parameters
    ----------
    initializer : Initializer
        Population initializer.
    name : str, optional
        Display name (default ``"RandomSearch"``).
    **kwargs
        Forwarded to :class:`HillClimb`.
    """

    def __init__(self, initializer: Initializer, name="RandomSearch", random_state=None, **kwargs):
        super().__init__(initializer=initializer, operator=create_operator(method="random.random", random_state=random_state), name=name, **kwargs)
