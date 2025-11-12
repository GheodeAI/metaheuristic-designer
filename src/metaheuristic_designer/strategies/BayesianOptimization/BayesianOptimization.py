from __future__ import annotations
from copy import copy
from ...SearchStrategy import SearchStrategy
from ...Initializer import Initializer
from ...ParamScheduler import ParamScheduler
from ...selectionMethods import SurvivorSelection, SurvivorSelectionNull, ParentSelection
from ...operators.BOOperator import BOOperator


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
            initializer, operator=BOOperator(params=params), parent_sel=parent_sel, survivor_sel=SurvivorSelectionNull(), params=params, name=name
        )
