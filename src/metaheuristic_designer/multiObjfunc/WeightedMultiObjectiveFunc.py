from __future__ import annotations
import numpy as np
from typing import Iterable
from ..ObjectiveFunc import *
from ..Population import Population


class WeightedMultiObjectiveFunc(ObjectiveFunc):
    def __init__(
        self,
        objectives: Iterable[ObjectiveFunc],
        weights: Iterable[float] = None,
        name: str = "Weighted average of functions",
    ):
        super().__init__(name=name)

        self.objectives = objectives

        if weights is None:
            weights = np.fill(1 / len(self.objectives), len(self.objectives))
        if sum(weights) != 1:
            weights = np.asarray(weights) / sum(weights)
        self.weights = weights

    def fitness(self, population: Population, adjusted=True) -> float:
        fitness_total = 0
        for obj_func, w in zip(self.objectives, self.weights):
            fitness_total += w * obj_func.fitness(population, adjusted)

        return fitness_total

    def objective(self, solution: Any) -> ndarray:
        # obj_total = 0
        # for obj_func, w in zip(self.objectives, self.weights):
        #     obj_total += w * obj_func.objective(solution)
        # return obj_total

        return np.array([objfunc.objective(solution) for objfunc in self.objectives])

    def repair_solution(self, solution: Any) -> Any:
        return solution
