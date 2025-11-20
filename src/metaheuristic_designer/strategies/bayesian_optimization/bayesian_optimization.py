from __future__ import annotations
from copy import copy
from ...search_strategy import SearchStrategy
from ...initializer import Initializer
from ...param_scheduler import ParamScheduler
from ...selection_methods import SurvivorSelection, NullSurvivorSelection, ParentSelection
from ...operators.BO_operator import BOOperator


class BayesianOptimization(SearchStrategy):
    """
    Bayesian Optimization
    """

    def __init__(
        self,
        initializer: Initializer,
        parent_sel: ParentSelection = None,
        params: ParamScheduler | dict = None,
        name: str = "Bayesian Optimization",
    ):
        super().__init__(
            initializer, operator=BOOperator(params=params), parent_sel=parent_sel, survivor_sel=NullSurvivorSelection(), params=params, name=name
        )
