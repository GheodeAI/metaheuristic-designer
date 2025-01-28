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
        modes: Iterable[str] = None,
        name: str = "Weighted average of functions",
    ):
        super().__init__(name=name)

        self.n_objectives = n_objectives
        self.modes = modes
        self.factors = np.ones(n_objectives)
        if modes is not None:
            if isinstance(modes, str) and modes == "min":
                self.factors *= -1
            else:
                self.factors[[i != "max" for i in modes]] = -1

    def fitness(self, population: Population, adjusted: bool = True) -> ndarray:
        self.counter += 1
        solutions = population.decode()
        values = self.objective(solutions)

        if adjusted:
            values = self.factors * (value - self.penalize(solutions))

        return values

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


class MultiObjectiveVectorFunc(MultiObjectiveFunc):
    def __init__(
        self,
        n_objectives: int,
        vecsize: int,
        low_lim: float = -100,
        up_lim: float = 100,
        modes: Iterable[str] = None,
        name: str = "Weighted average of functions",
    ):
        super().__init__(n_objectives=n_objectives, modes=modes, name=name)

        self.vecsize = vecsize
        self.low_lim = low_lim
        self.up_lim = up_lim

    def repair_solution(self, vector: ndarray) -> ndarray:
        return np.clip(vector, self.low_lim, self.up_lim)