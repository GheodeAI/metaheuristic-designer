from __future__ import annotations
from typing import Optional
from ..initializer import Initializer
from ..search_strategy import SearchStrategy
from ..operator import Operator
from ..population import Population
from ..survivor_selection import SurvivorSelection
from ..survivor_selection_methods import create_survivor_selection
from ..utils import check_random_state, RNGLike


class LocalSearch(SearchStrategy):
    """
    Local search algorithm
    """

    def __init__(
        self,
        initializer: Initializer,
        operator: Optional[Operator] = None,
        survivor_sel: Optional[SurvivorSelection] = None,
        name: str = "LocalSearch",
        iterations: int = 100,
        random_state: Optional[RNGLike] = None,
        **kwargs,
    ):
        if survivor_sel is None:
            survivor_sel = create_survivor_selection("many-to-one")

        super().__init__(
            initializer,
            operator=operator,
            survivor_sel=survivor_sel,
            name=name,
            random_state=random_state,
            # Forced kwargs
            iterations=iterations,
            **kwargs,
        )

    def perturb(self, parents: Population, **kwargs) -> Population:
        new_population = parents.repeat(self.params.iterations)
        return super().perturb(new_population, **kwargs)
