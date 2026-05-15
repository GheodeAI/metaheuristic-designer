"""
Module for recording per-generation metrics and exporting them as pandas DataFrames.
"""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .algorithm import Algorithm

logger = logging.getLogger(__name__)


class HistoryTracker:
    """Record per-generation metrics and export them as pandas DataFrames.

    The tracker is called once per generation (via :meth:`step`) and
    stores the requested statistics.  After the run the data can be
    retrieved with :meth:`to_pandas` (a summary of best, median, worst,
    diversity, and scheduled parameters) or :meth:`to_pandas_full_objective`
    (the full objective vector of every individual at each generation).

    Parameters
    ----------
    track_best : bool, optional
        Record the best objective and solution (default ``True``).
    track_median : bool, optional
        Record the median objective (default ``False``).
    track_worst : bool, optional
        Record the worst objective (default ``False``).
    track_full_objective : bool, optional
        Store the complete objective vector of the population at every
        generation.  Enables :meth:`to_pandas_full_objective`.
    track_full_population : bool, optional
        Store the entire population (genotypes) at every generation.
        This can consume a lot of memory.
    track_parameters : bool, optional
        Record the current value of all scheduled parameters (e.g.,
        mutation strength, branch probability).
    track_diversity : bool, optional
        Compute and store a simple diversity metric (average
        Euclidean distance from the centroid).
    """

    def __init__(
        self,
        track_best=True,
        track_median=False,
        track_worst=False,
        track_full_objective=False,
        track_full_population=False,
        track_parameters=False,
        track_diversity=False,
    ):
        self.track_best = track_best
        self.track_median = track_median
        self.track_worst = track_worst
        self.track_full_objective = track_full_objective
        self.track_full_population = track_full_population
        self.track_diversity = track_diversity
        self.track_parameters = track_parameters

        self.best_solutions = []
        self.median_solutions = []
        self.worst_solutions = []

        self.best_objective = []
        self.median_objective = []
        self.worst_objective = []

        self.complete_population = []
        self.complete_objective = []
        self.diversity = []
        self.parameters = []

        self.recorded_iterations = []

    def restart(self):
        """Clear all recorded data.

        Call this when an algorithm is reset to start a fresh run.
        """

        self.best_solutions = []
        self.median_solutions = []
        self.worst_solutions = []

        self.best_objective = []
        self.median_objective = []
        self.worst_objective = []

        self.complete_population = []
        self.complete_objective = []
        self.diversity = []
        self.parameters = []

        self.recorded_iterations = []

    def step(self, algorithm: Algorithm):
        """Record metrics for the current generation.

        Parameters
        ----------
        algorithm : Algorithm
            The running algorithm from which the current population,
            fitness, objective, and parameters are extracted.
        """

        population = algorithm.population
        solutions = population.decode()
        fitness_array = population.fitness
        objective_array = population.objective
        fitness_order = np.argsort(fitness_array)

        self.recorded_iterations.append(algorithm.stopping_condition.iterations)

        if self.track_full_objective:
            self.complete_objective.append(objective_array)

        if self.track_full_population:
            self.complete_population.append(solutions)

        if self.track_best:
            best_idx = fitness_order[-1]
            self.best_solutions.append(solutions[best_idx])
            self.best_objective.append(objective_array[best_idx])

        if self.track_median:
            half_size = len(fitness_array) // 2
            if len(fitness_array) % 2 == 0:
                median_idx = fitness_order[half_size - 1]
            else:
                median_idx = fitness_order[half_size]
            self.median_solutions.append(solutions[median_idx])
            self.median_objective.append(objective_array[median_idx])

        if self.track_worst:
            worst_idx = fitness_order[0]
            self.worst_solutions.append(solutions[worst_idx])
            self.worst_objective.append(objective_array[worst_idx])

        if self.track_diversity:
            # WIP, right now we have an basic euclidean distance based metric, more flexible methods expected for next versions
            genotype_matrix = algorithm.population.genotype_matrix
            centroid = np.mean(genotype_matrix, axis=0)
            dists = np.sqrt(np.sum((genotype_matrix - centroid) ** 2, axis=1))

            self.diversity.append(np.mean(dists))

        if self.track_parameters:
            self.parameters.append(algorithm.gather_parameters())

    def to_pandas(self):
        """Return a DataFrame with per-generation summary metrics.

        Columns include ``iteration``, ``best_objective``,
        ``median_objective``, ``worst_objective``, ``diversity``,
        and one column per scheduled parameter.  The DataFrame is
        intended for easy plotting with seaborn or matplotlib.

        Returns
        -------
        pandas.DataFrame
        """

        data_dict = {"iteration": np.asarray(self.recorded_iterations)}

        if self.track_best:
            data_dict["best_objective"] = self.best_objective

        if self.track_median:
            data_dict["median_objective"] = self.median_objective

        if self.track_worst:
            data_dict["worst_objective"] = self.worst_objective

        if self.track_diversity:
            data_dict["diversity"] = self.diversity

        if self.track_parameters:
            # Let pandas do the data reordering and then get back a dictionary
            param_df = pd.DataFrame(self.parameters)
            data_dict.update(param_df.to_dict(orient="list"))

        return pd.DataFrame.from_dict(data_dict)

    def to_pandas_full_objective(self):
        """Return a wide-format DataFrame of all individual objective values.

        Each column ``Individual_0``, ``Individual_1``, ... holds the
        objective of one member of the population across generations.
        This is useful for boxplots or distribution plots of fitness.

        Returns
        -------
        pandas.DataFrame
            Empty DataFrame if *track_full_objective* was not enabled.
        """

        if not self.track_full_objective:
            logger.warning("Tried to extract the full objective history but it was not being tracked.")
            return pd.DataFrame()

        data_dict = {"iteration": np.asarray(self.recorded_iterations)}
        complete_objective_arr = np.asarray(self.complete_objective)
        data_dict.update(
            {f"Individual_{idx:d}": objective_values for idx, objective_values in enumerate(complete_objective_arr.T)}  # Iterates for each individual
        )

        return pd.DataFrame.from_dict(data_dict)

    def get_state(self):
        """Return a dictionary containing the recorded history.

        Returns
        -------
        dict
            Keys include ``best_objective``, ``best_solutions``, etc.
            Only metrics that were enabled are present.
        """

        data = {
            "class_name": self.__class__.__name__,
        }

        if self.track_best:
            data["best_solutions"] = self.best_solutions
            data["best_objective"] = self.best_objective

        if self.track_median:
            data["median_solutions"] = self.median_solutions
            data["median_objective"] = self.median_objective

        if self.track_worst:
            data["worst_solutions"] = self.worst_solutions
            data["worst_objective"] = self.worst_objective

        if self.complete_objective:
            data["complete_objectives"] = self.complete_objective

        if self.track_full_population:
            data["populations"] = self.complete_population

        if self.track_diversity:
            data["divesity"] = self.diversity

        return data
