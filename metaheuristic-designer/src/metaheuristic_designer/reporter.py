from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from metaheuristic_designer.algorithm import Algorithm


class Reporter(ABC):
    def __init__(self, **kwargs):
        """_summary_"""

    @abstractmethod
    def log_init(self, algorithm: Algorithm):
        """_summary_

        Parameters
        ----------
        algorithm : Algorithm
            _description_
        """

    @abstractmethod
    def log_step(self, algorithm: Algorithm):
        """_summary_

        Parameters
        ----------
        algorithm : Algorithm
            _description_
        """

    @abstractmethod
    def log_end(self, algorithm: Algorithm):
        """_summary_

        Parameters
        ----------
        algorithm : Algorithm
            _description_
        """
