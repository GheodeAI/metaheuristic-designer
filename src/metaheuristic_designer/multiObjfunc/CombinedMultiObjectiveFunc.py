from __future__ import annotations
import numpy as np
from numpy import ndarray
from typing import Iterable
from .MultiObjectiveFunc import MultiObjectiveFunc


class CombinedMultiObjectiveFunc(MultiObjectiveFunc):
    def __init__(
        self,
        objectives: List[ObjectiveFunc],
        weights: Iterable[float] = None,
        name: str = "Combined functions",
    ):
        super().__init__(name=name)

        self.objectives = objectives

    def fitness(self, indiv: Individual, adjusted=True) -> ndarray:
        return np.array([objfunc.fitness(indiv, adjusted) for i in self.objectives])

    def objective(self, solution: Any) -> ndarray:
        return np.array([objfunc.objective(solution) for i in self.objectives])
