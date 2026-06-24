"""
Hill Climbing strategy (single-solution, greedy local improvement).
"""

from __future__ import annotations
from typing import Optional
from ...initializer import Initializer
from ...survivor_selection_base import SurvivorSelection
from ...search_strategy import SearchStrategy
from ...operator import Operator
from ...survivor_selection import create_survivor_selection
from ...utils import RNGLike
from ..single_solution_strategy import SingleSolutionStrategy


class HillClimb(SingleSolutionStrategy):
    """
    Hill Climbing algorithm.

    A single solution is perturbed each iteration.  If the new
    solution is better, it replaces the current one.  By default,
    the survivor selection is set to one-to-one competition
    (``"hill_climb"`` in the survivor registry).

    Parameters
    ----------
    initializer : Initializer
        Population initializer (typically creates a single individual).
    operator : Operator, optional
        Perturbation operator.  Defaults to :class:`NullOperator`.
    survivor_sel : SurvivorSelection, optional
        Survivor selection method; defaults to ``"hill_climb"``.
    params : dict, optional
        Additional parameters stored as schedulable values.
    name : str, optional
        Display name (default ``"HillClimb"``).
    rng : RNGLike, optional
        Random number generator.
    \\*\\*kwargs
        Forwarded to :class:`SearchStrategy`.
    """

    def __init__(
        self,
        initializer: Initializer,
        operator: Optional[Operator] = None,
        survivor_sel: Optional[SurvivorSelection] = None,
        params: Optional[dict] = None,
        name: str = "HillClimb",
        rng: Optional[RNGLike] = None,
        **kwargs,
    ):

        if survivor_sel is None:
            survivor_sel = create_survivor_selection("hill_climb")

        super().__init__(initializer, operator=operator, survivor_sel=survivor_sel, params=params, name=name, rng=rng, **kwargs)
