from __future__ import annotations
from abc import abstractmethod
import numpy as np
from numpy import ndarray
from typing import Iterable
from ..ObjectiveFunc import *


class MultiObjectiveFunc(ObjectiveFunc):
    def __init__(
        self,
        n_objectives: int,
        name: str = "Weighted average of functions",
    ):
        super().__init__(name=name)

        self.n_objectives = n_objectives

    @abstractmethod
    def fitness(self, indiv: Individual, adjusted=True) -> ndarray:
        self.counter += 1
        solution = indiv.encoding.decode(indiv.genotype)
        value = self.objective(solution)

        if adjusted:
            value = value - self.penalize(solution)

        return value

    @abstractmethod
    def objective(self, solution: Any) -> ndarray:
        """"""

    def penalize(self, solution: Any) -> ndarray:
        """
        Gives a penalization to the fitness value of an individual if it violates any constraints propotional
        to how far it is to a viable solution.

        If not implemented always returns 0.

        Parameters
        ----------
        solution: Any
            A solution that could be violating the restrictions of the problem.

        Returns
        -------
        penalty: float
            The penalty associated to the degree that the solution violates the restrictions of the problem.
        """

        return np.zeros(self.n_objectives)
