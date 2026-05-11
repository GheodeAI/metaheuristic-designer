"""
Random search strategy (baseline).
"""

from __future__ import annotations
from ...operators import create_operator
from .hill_climb import HillClimb


class RandomSearch(HillClimb):
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

    def __init__(self, initializer, name="RandomSearch", **kwargs):
        super().__init__(initializer, operator=create_operator(method="random.random"), name=name, **kwargs)
