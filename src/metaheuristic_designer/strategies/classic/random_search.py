"""
Random search strategy (baseline).
"""

from __future__ import annotations

from ...initializer import Initializer
from ...operators import create_operator
from ..population_based_strategy import PopulationBasedStrategy


class RandomSearch(PopulationBasedStrategy):
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

    def __init__(self, initializer: Initializer, name="RandomSearch", rng=None, **kwargs):
        super().__init__(initializer=initializer, operator=create_operator(method="random.random", initializer=initializer, rng=rng), name=name, **kwargs)
