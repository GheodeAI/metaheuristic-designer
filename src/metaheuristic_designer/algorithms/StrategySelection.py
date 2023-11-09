from __future__ import annotations
import time
import pandas as pd
from matplotlib import pyplot as plt
from ..SearchStrategy import SearchStrategy
from .GeneralAlgorithm import GeneralAlgorithm
from .AlgorithmSelection import AlgorithmSelection
from collections import Counter


class StrategySelection:
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

    def __init__(
        self,
        objfunc: ObjectiveFunc,
        strategy_list: Iterable[SearchStrategy],
        algorithm_params: Union[ParamScheduler, dict] = None,
        params: Union[ParamScheduler, dict] = None,
    ):
        self.strategy_list = strategy_list
        algorithm_list = [GeneralAlgorithm(objfunc, strategy, algorithm_params) for strategy in strategy_list]

        self.algorithm_selection = AlgorithmSelection(algorithm_list, params)

    def optimize(self) -> Tuple[Any, float, pd.DataFrame]:
        """
        Evaluates all the provided search strategies and returns the best overall solution
        """

        return self.algorithm_selection.optimize()
