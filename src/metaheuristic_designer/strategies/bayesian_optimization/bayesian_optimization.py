from __future__ import annotations
from copy import copy
from ...search_strategy import SearchStrategy
from ...initializer import Initializer
from ...parent_selection import ParentSelection
from ...operators.BO_operator import BOOperator


class BayesianOptimization(SearchStrategy):
    """
    Bayesian Optimization
    """

    def __init__(self, initializer: Initializer, parent_sel: ParentSelection = None, name: str = "Bayesian Optimization", **kwargs):
        super().__init__(initializer, operator=BOOperator(**kwargs), parent_sel=parent_sel, name=name, **kwargs)
