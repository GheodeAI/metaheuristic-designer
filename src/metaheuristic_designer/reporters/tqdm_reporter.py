"""
Reporter that shows a tqdm progress bar during optimization.
"""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING
from math import floor
from tqdm.auto import tqdm
from ..reporter import Reporter

if TYPE_CHECKING:
    from metaheuristic_designer.algorithm import Algorithm

logger = logging.getLogger(__name__)


class TQDMReporter(Reporter):
    """Reporter that displays a tqdm progress bar.

    Parameters
    ----------
    resolution : int, optional
        Number of ticks in the progress bar (default 1000).  Higher
        values give smoother updates.
    """

    def __init__(self, resolution: int = 1000, **kwargs):
        if not isinstance(resolution, int):
            resolution = int(resolution)
            logger.warning("Implicitly converted progress value to int.")
        self.resolution = resolution
        self.rounded_progress = 0
        self.bar_tracker = None

    def log_init(self, algorithm: Algorithm):
        """Initialise the progress bar and display the first postfix."""
        objfunc_name = algorithm.objfunc.name
        alg_name = algorithm.name

        self.bar_tracker = tqdm(total=self.resolution, bar_format="{l_bar}{bar}| {percentage:3.0f}% [{elapsed}<{remaining}, {rate_fmt}{postfix}]")
        self.bar_tracker.set_description(f"Optimizing {objfunc_name} using {alg_name}, Iteration 0")
        self.bar_tracker.set_postfix(evals=0)

        self.rounded_progress = 0

    def log_step(self, algorithm: Algorithm):
        """Update the progress bar with current iteration, evaluations, and fitness."""
        clipped_progress = min(max(0, algorithm.progress), 1)
        next_rounded_progress = floor(clipped_progress * self.resolution)

        objfunc_name = algorithm.objfunc.name
        alg_name = algorithm.name
        iterations = algorithm.iterations
        evaluations = algorithm.stopping_condition.evaluations
        _, best_objective = algorithm.best_solution()

        self.bar_tracker.set_description(f"Optimizing {objfunc_name} using {alg_name}, Iteration {iterations:,}")
        self.bar_tracker.set_postfix(evals=f"{evaluations:,}", fitness=f"{best_objective:.6g}")

        if next_rounded_progress > self.rounded_progress:
            self.bar_tracker.update(next_rounded_progress - self.rounded_progress)
            self.rounded_progress = next_rounded_progress

    def log_end(self, algorithm: Algorithm):
        """Fill the progress bar to 100% and close it."""
        objfunc_name = algorithm.objfunc.name
        alg_name = algorithm.name
        iterations = algorithm.iterations
        evaluations = algorithm.stopping_condition.evaluations
        remaining = self.resolution - self.rounded_progress
        _, best_objective = algorithm.best_solution()

        self.bar_tracker.set_description(f"Done optimizing {objfunc_name} using {alg_name}, Iteration {iterations}")
        self.bar_tracker.set_postfix(evals=evaluations, fitness=best_objective)

        if remaining > 0:
            self.bar_tracker.update(remaining)

        self.bar_tracker.close()
