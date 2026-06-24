"""
Convenience wrapper that benchmarks strategies instead of pre-built algorithms.
"""

from __future__ import annotations
from typing import Iterable
from ..search_strategy import SearchStrategy
from ..objective_function import ObjectiveFunc
from .algorithm_selection import AlgorithmSelection
from ..algorithm import Algorithm


class StrategySelection(AlgorithmSelection):
    """Evaluate a set of search strategies by automatically wrapping them in
    :class:`Algorithm` objects.

    This is a thin wrapper around :class:`AlgorithmSelection` that accepts
    :class:`SearchStrategy` instances and a shared configuration dictionary.
    It converts each strategy into an :class:`Algorithm` with the same
    settings and then delegates to the parent class.

    Parameters
    ----------
    objfunc : ObjectiveFunc
        Objective function to evaluate.
    strategy_list : iterable of SearchStrategy
        The search strategies to compare.
    repetitions : int, optional
        Number of independent runs per strategy (default 10).
    \\*\\*kwargs
        Keyword arguments forwarded to every :class:`Algorithm` constructor
        (e.g., ``stop_cond="max_iterations"``, ``max_iterations=100``).
    """

    def __init__(self, objfunc: ObjectiveFunc, strategy_list: Iterable[SearchStrategy], repetitions: int = 10, **kwargs):
        self.strategy_list = strategy_list
        algorithm_list = [Algorithm(objfunc, strategy, **kwargs) for strategy in strategy_list]

        super().__init__(algorithm_list=algorithm_list, repetitions=repetitions)
