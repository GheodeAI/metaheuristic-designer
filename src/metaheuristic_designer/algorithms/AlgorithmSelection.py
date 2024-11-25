from __future__ import annotations
from typing import Iterable, Tuple, Any
from collections import Counter
import pandas as pd
from matplotlib import pyplot as plt
import enlighten
from ..Algorithm import Algorithm
from ..ParamScheduler import ParamScheduler


class AlgorithmSelection:
    """
    Utility to evaluate and compare the performance of different algorithms.

    Parameters
    ----------

    algorithm_list: Iterable[Algorithm]
        List of algorithms to evaluate.
    params: ParamScheduler or dict, optional
        Indicates whether to show progress bars with 'verbose' and the number of times to repeat each algorithm with 'repetitions'
    """

    def __init__(
        self,
        algorithm_list: Iterable[Algorithm],
        params: ParamScheduler | dict = None,
    ):
        if params is None:
            params = {}

        self.repetitions = params.get("repetitions", 10)

        self.algorithm_list = algorithm_list

        # Avoid repeating names
        name_counter = Counter()
        for alg in algorithm_list:
            prev_name = alg.name
            if alg.name in name_counter:
                alg.name = alg.name + str(name_counter[alg.name] + 1)
            name_counter.update([prev_name])

        self.solutions = []
        self.verbose = params.get("verbose", True)

    def optimize(self) -> Tuple[Any, float, pd.DataFrame]:
        """
        Evaluates all the provided search strategies and returns the best overall solution
        """

        if self.verbose:
            print(f"Running {len(self.algorithm_list)} algorithms {self.repetitions} times each.")

        best_solution = None
        best_fitness = 0
        report_raw = pd.DataFrame(columns=["name", "realtime", "cputime", "fitness"])

        # Create progress bar manager and global progress bar
        if self.verbose:
            bar_manager = enlighten.get_manager()
            algorithm_bar = bar_manager.counter(total=len(self.algorithm_list), desc="Launching algorithms", color="red")

        for algorithm in self.algorithm_list:
            # Create new progress bar for the new algorithm
            if self.verbose:
                repetition_bar = bar_manager.counter(
                    total=self.repetitions,
                    desc=f"Evaluating {algorithm.name}",
                    color="green",
                    leave=False,
                )

            for _ in range(self.repetitions):
                # Optimize using the algorithm
                population = algorithm.optimize()
                solution, fitness = population.best_solution()

                # Get the dataframe row
                report_raw.loc[len(report_raw.index)] = {
                    "name": algorithm.name,
                    "realtime": algorithm.real_time_spent,
                    "cputime": algorithm.cpu_time_spent,
                    "fitness": fitness,
                }

                # Save the solution if it improves the previous one
                if (
                    best_solution is None
                    or (algorithm.objfunc.mode == "min" and best_fitness > fitness)
                    or (algorithm.objfunc.mode == "max" and best_fitness < fitness)
                ):
                    best_solution = solution
                    best_fitness = fitness

                # Update progress bar
                if self.verbose:
                    repetition_bar.update()

                # Reset the algorithm data
                algorithm.restart()

            # Update progress bar
            if self.verbose:
                algorithm_bar.update()

        # Stop displaying progress bars
        if self.verbose:
            bar_manager.stop()

        # Obtain statistics about the executions
        report_gropued = report_raw.groupby("name", sort=False)
        report = pd.DataFrame()
        for group_name, group in report_gropued:
            report = pd.concat(
                [
                    report,
                    pd.DataFrame(
                        {
                            "name": [group_name],
                            "realtime_min": [group["realtime"].min()],
                            "realtime_avg": [group["realtime"].mean()],
                            "realtime_max": [group["realtime"].max()],
                            "realtime_std": [group["realtime"].std()],
                            "cputime_min": [group["cputime"].min()],
                            "cputime_avg": [group["cputime"].mean()],
                            "cputime_max": [group["cputime"].max()],
                            "cputime_std": [group["cputime"].std()],
                            "fitness_min": [group["fitness"].min()],
                            "fitness_avg": [group["fitness"].mean()],
                            "fitness_max": [group["fitness"].max()],
                            "fitness_std": [group["fitness"].std()],
                        }
                    ),
                ]
            )

        return best_solution, best_fitness, report.reset_index(drop=True)
