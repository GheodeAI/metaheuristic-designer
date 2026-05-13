from __future__ import annotations
from typing import Optional
from ..initializer import Initializer
from ..survivor_selection_base import SurvivorSelection
from ..search_strategy import SearchStrategy
from ..operator import Operator
from ..survivor_selection import create_survivor_selection
from ..utils import check_random_state, RNGLike


class HillClimb(SearchStrategy):
    """
    Hill Climbing algorithm
    """

    def __init__(
        self,
        initializer: Initializer,
        operator: Optional[Operator] = None,
        survivor_sel: Optional[SurvivorSelection] = None,
        params: Optional[dict] = None,
        name: str = "HillClimb",
        random_state: Optional[RNGLike] = None,
        **kwargs,
    ):

        if survivor_sel is None:
            survivor_sel = create_survivor_selection("hill_climb")

        super().__init__(initializer, operator=operator, survivor_sel=survivor_sel, params=params, name=name, random_state=random_state, **kwargs)
