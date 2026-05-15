"""
Utility for benchmarking a set of algorithms with independent repetitions.
"""

from __future__ import annotations
from copy import copy
from typing import Iterable, Tuple
from collections import Counter
import pandas as pd
from tqdm.auto import tqdm

from ..history_tracker import HistoryTracker
from ..reporters import SilentReporter
from ..population import Population
from ..algorithm import Algorithm


class AlgorithmSelection:
    """Run several algorithms multiple times and collect performance data.

    Each algorithm in the list is executed ``repetitions`` times.
    During the runs, every algorithm gets a silent reporter and a
    fresh :class:`HistoryTracker` that records best, median, and
    worst objectives.  After all runs finish, you can obtain the
    raw per-repetition data (:attr:`raw_data`) and an aggregated
    :meth:`report`.

    Parameters
    ----------
    algorithm_list : iterable of Algorithm
        The algorithms to evaluate.  They are copied before
        execution so the originals are not modified.
    repetitions : int, optional
        How many independent repetitions to perform for each
        algorithm (default 10).
    """

    def __init__(self, algorithm_list: Iterable[Algorithm], repetitions: int = 10):
        self.repetitions = repetitions

        self.display_names = []
        self.algorithm_list = []
        name_counter = Counter()
        for alg in algorithm_list:
            prev_name = alg.name
            if alg.name in name_counter:
                self.display_names.append(f"{alg.name}{name_counter[alg.name] + 1}")
            else:
                self.display_names.append(alg.name)
            name_counter.update([prev_name])
            alg_copy = copy(alg)
            alg_copy.reporter = SilentReporter()
            alg_copy.history_tracker = HistoryTracker(track_best=True, track_median=True, track_worst=True)
            self.algorithm_list.append(alg_copy)

        self.opt_mode = algorithm_list[0].objfunc.mode

    def optimize(self) -> Tuple[Population, pd.DataFrame]:
        """Execute all repetitions and return the best population found.

        The raw data is stored in :attr:`self.raw_data` for later
        inspection.

        Returns
        -------
        Population
            The population with the best overall fitness across all
            repetitions and algorithms.
        """

        best_fitness = 0
        best_population = None

        raw_rows = []
        outer_bar = tqdm(total=len(self.algorithm_list), desc="Algorithms", position=0, leave=True)
        for alg_idx, algorithm in enumerate(self.algorithm_list):
            inner_bar = tqdm(total=self.repetitions, desc=f"  {algorithm.name}", position=1, leave=False)
            for rep_i in range(self.repetitions):
                # Optimize using the algorithm
                population = algorithm.optimize()
                _, fitness = population.best_individual()

                # Get the dataframe row
                stop_cond = algorithm.stopping_condition
                history_tracker = algorithm.history_tracker
                raw_rows.append(
                    {
                        "repetition": rep_i,
                        "name": self.display_names[alg_idx],
                        "iterations": stop_cond.iterations,
                        "evaluations": stop_cond.evaluations,
                        "realtime": stop_cond.real_time_spent,
                        "cputime": stop_cond.cpu_time_spent,
                        "best_objective": history_tracker.best_objective[-1],
                        "median_objective": history_tracker.median_objective[-1],
                        "worst_objective": history_tracker.worst_objective[-1],
                    }
                )

                # Save the solution if it improves the previous one
                if best_population is None or fitness > best_fitness:
                    best_fitness = fitness
                    best_population = population

                # Reset the algorithm data
                algorithm.restart()
                inner_bar.update(1)
            inner_bar.close()
            outer_bar.update(1)
        outer_bar.close()

        # Obtain statistics about the executions
        self.raw_data = pd.DataFrame.from_dict(raw_rows)

        return best_population

    def report(self) -> pd.DataFrame:
        """Return an aggregated summary of the raw data.

        The report contains one row per algorithm, with columns for
        the number of runs, overall best, average best, standard
        deviation, timing averages, and more.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the aggregated statistics.
        """

        return (
            self.raw_data.groupby("name")
            .agg(
                runs=("best_objective", "count"),
                overall_best=("best_objective", self.opt_mode),
                avg_best=("best_objective", "mean"),
                std_best=("best_objective", "std"),
                avg_realtime=("realtime", "mean"),
                avg_cputime=("cputime", "mean"),
                avg_iterations=("iterations", "mean"),
                avg_evaluations=("evaluations", "mean"),
                median_of_medians=("median_objective", "median"),  # median of the per‑run medians
                avg_worst=("worst_objective", "mean"),  # average worst fitness
                overall_worst=("worst_objective", "min" if self.opt_mode == "max" else "max"),
            )
            .reset_index()
        )
