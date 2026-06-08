"""
Local Search strategy (single solution, multiple perturbations per iteration).
"""

from __future__ import annotations
from typing import Optional

from ...parent_selection import create_parent_selection
from ...initializer import Initializer
from ...search_strategy import SearchStrategy
from ...operator import Operator
from ...population import Population
from ...survivor_selection_base import SurvivorSelection
from ...survivor_selection import create_survivor_selection
from ...utils import RNGLike
from ..population_based_strategy import PopulationBasedStrategy


class LocalSearch(PopulationBasedStrategy):
    """
    Local Search algorithm.

    At each iteration the current solution is duplicated *iterations*
    times, and every copy is perturbed independently.  The best
    among the original and the perturbed copies survives.  By default,
    the survivor selection is set to ``"local_search"`` (one parent
    vs. many offspring).

    Parameters
    ----------
    initializer : Initializer
        Population initializer.
    operator : Operator, optional
        Perturbation operator.
    survivor_sel : SurvivorSelection, optional
        Survivor selection; defaults to ``"local_search"``.
    name : str, optional
        Display name (default ``"LocalSearch"``).
    iterations : int, optional
        Number of perturbed copies per iteration (default 100).
    rng : RNGLike, optional
        Random number generator.
    **kwargs
        Forwarded to :class:`SearchStrategy`.
    """

    def __init__(
        self,
        initializer: Initializer,
        operator: Optional[Operator] = None,
        survivor_sel: Optional[SurvivorSelection] = None,
        name: str = "LocalSearch",
        iterations: int = 100,
        rng: Optional[RNGLike] = None,
        **kwargs,
    ):
        if survivor_sel is None:
            survivor_sel = create_survivor_selection("local_search")

        super().__init__(
            initializer,
            parent_sel=create_parent_selection("repeat", amount=iterations * initializer.population_size),
            operator=operator,
            survivor_sel=survivor_sel,
            name=name,
            rng=rng,
            # Forced kwargs
            iterations=iterations,
            **kwargs,
        )
