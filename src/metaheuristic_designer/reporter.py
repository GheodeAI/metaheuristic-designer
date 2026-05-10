"""
Module defining the abstract reporter interface for algorithm progress output.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from metaheuristic_designer.algorithm import Algorithm


class Reporter(ABC):
    """Abstract interface for progress reporters.

    A reporter is notified at three key moments of an optimisation
    run: initialisation, after each generation, and at completion.
    Implementations can display progress bars, log messages, update
    dashboards, etc.
    """

    @abstractmethod
    def log_init(self, algorithm: Algorithm):
        """Called once, before the main optimisation loop starts.

        Parameters
        ----------
        algorithm : Algorithm
            The algorithm that is about to run.
        """

    @abstractmethod
    def log_step(self, algorithm: Algorithm):
        """Called after each generation.

        Parameters
        ----------
        algorithm : Algorithm
            The running algorithm, with up-to-date population, iteration
            count, and best solution.
        """

    @abstractmethod
    def log_end(self, algorithm: Algorithm):
        """Called once, after the optimisation loop finishes.

        Parameters
        ----------
        algorithm : Algorithm
            The algorithm that has just finished.
        """
