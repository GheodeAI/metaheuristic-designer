from __future__ import annotations
import time
import pandas as pd
from matplotlib import pyplot as plt
from ..Algorithm import Algorithm
from collections import Counter


class AlgorithmSelection:
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
        algorithm_list: Iterable[Algorithm],
        params: Union[ParamScheduler, dict] = None,
    ):
        if params is None:
            params = {}

        self.algorithm_list = algorithm_list

        # Avoid repeating names
        name_counter = Counter()
        for alg in algorithm_list:
            prev_name = alg.name
            if alg.name in name_counter:
                alg.name = alg.name + str(name_counter[alg.name] + 1)
            name_counter.update([prev_name])

        self.solutions = []
        self.report = pd.DataFrame(columns=["name","generations","evaluations","real time","CPU time","fitness","solution"])
        self.verbose = params.get("verbose", True)

    def optimize(self):
        if self.verbose:
            print(f"Running {len(self.algorithm_list)} algorithms.")
        
        for idx, algorithm in enumerate(self.algorithm_list):
            best_solution, best_fitness = algorithm.optimize()
            self.report.loc[len(self.report.index)] = {
                "name": algorithm.name,
                "generations": algorithm.steps+1,
                "evaluations": algorithm.objfunc.counter,
                "real time": algorithm.real_time_spent,
                "CPU time": algorithm.cpu_time_spent,
                "fitness": best_fitness,
                "solution": str(best_solution)
            }

            if self.verbose:
                print(f"{algorithm.name} finished. {idx+1}/{len(self.algorithm_list)}")

            algorithm.restart()
        
        self.report = self.report.sort_values("fitness")
        
        return self.report
