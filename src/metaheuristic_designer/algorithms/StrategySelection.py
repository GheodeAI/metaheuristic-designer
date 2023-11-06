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
    General framework for metaheuristic algorithms

    Parameters
    ----------

    objfunc: ObjectiveFunc
        Objective function to be optimized.
    search_strategy: Algorithm
        Search strategy that will iteratively optimize the function.
    params: ParamScheduler or dict, optional
        Dictionary of parameters to define the stopping condition and output of the algorithm.
    """

    def __init__(
        self,
        objfunc: ObjectiveFunc,
        strategy_list: Iterable[SearchStrategy],
        params: Union[ParamScheduler, dict] = None,
    ):
        self.strategy_list = strategy_list
        algorithm_list = [GeneralAlgorithm(objfunc, strategy, params) for i in strategy_list]

        self.algorithm_selection = AlgorithmSelection(algorithm_list, params)

    def optimize(self):
        return self.algorithm_selection.optimize()
