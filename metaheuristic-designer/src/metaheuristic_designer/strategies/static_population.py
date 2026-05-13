from __future__ import annotations
from typing import Optional
from ..initializer import Initializer
from ..parent_selection_base import ParentSelection
from ..survivor_selection_base import SurvivorSelection
from ..search_strategy import SearchStrategy
from ..operator import Operator
from ..utils import RNGLike


class StaticPopulation(SearchStrategy):
    """
    Population-based algorithm where each individual is iteratively evolved with a given operator
    """

    def __init__(
        self,
        initializer: Initializer,
        operator: Operator,
        parent_sel: Optional[ParentSelection] = None,
        survivor_sel: Optional[SurvivorSelection] = None,
        name: str = "Static Population Evolution",
        random_state: Optional[RNGLike] = None,
        **kwargs,
    ):
        super().__init__(
            initializer, operator=operator, parent_sel=parent_sel, survivor_sel=survivor_sel, name=name, random_state=random_state, **kwargs
        )
