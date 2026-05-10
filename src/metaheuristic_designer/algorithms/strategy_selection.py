from __future__ import annotations
from typing import Iterable
from ..search_strategy import SearchStrategy
from ..objective_function import ObjectiveFunc
from .algorithm_selection import AlgorithmSelection
from ..algorithm import Algorithm


class StrategySelection(AlgorithmSelection):
    """
    Utility to evaluate and compare the performance of different search strategies.

    Parameters
    ----------

    objfunc: ObjectiveFunc
        Objective function to evaluate
    strategy_list: Iterable[SearchStrategy]
        List of algorithms to evaluate.
    algorithm_params: ParamScheduler or dict, optional
        Parameters shared by all the algorithms being run.
    params: ParamScheduler or dict, optional
        Indicates whether to show progress bars with 'verbose' and the number of times to repeat each algorithm with 'repetitions'
    """

    def __init__(self, objfunc: ObjectiveFunc, strategy_list: Iterable[SearchStrategy], repetitions: int = 10, **kwargs):
        self.strategy_list = strategy_list
        algorithm_list = [Algorithm(objfunc, strategy, **kwargs) for strategy in strategy_list]

        super().__init__(algorithm_list=algorithm_list, repetitions=repetitions)
