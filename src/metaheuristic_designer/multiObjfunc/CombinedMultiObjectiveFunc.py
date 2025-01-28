from __future__ import annotations
import numpy as np
from numpy import ndarray
from typing import Iterable
from .MultiObjectiveFunc import MultiObjectiveFunc


class CombinedMultiObjectiveFunc(MultiObjectiveFunc):
    def __init__(
        self,
        objectives: List[ObjectiveFunc],
        name: str = "Combined functions",
    ):
        super().__init__(name=name, n_objectives=len(objectives))

        self.objectives = objectives

    def fitness(self, population: population, adjusted=True) -> ndarray:
        return np.vstack([objfunc.fitness(population, adjusted) for objfunc in self.objectives]).T

    def objective(self, solution: Any) -> ndarray:
        return np.array([objfunc.objective(solution) for objfunc in self.objectives])
    
    def repair_solution(self, solution: Any) -> ndarray:
        for obj in self.objectives:
            solution = obj.repair_solution(solution)
        return solution
