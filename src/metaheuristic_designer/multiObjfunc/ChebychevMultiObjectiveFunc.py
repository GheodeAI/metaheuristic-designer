from __future__ import annotations
import numpy as np
from typing import Iterable
from ..ObjectiveFunc import *


class ChevychevMultiObjectiveFunc(ObjectiveFunc):
    """
    https://doi.org/10.1007/
    """
    
    def __init__(
        self,
        objectives: List[ObjectiveFunc],
        weights: Iterable[float] = None,
        epsilon: float = 1e-4,
        name: str = "Weighted average of functions",
    ):
        super().__init__(name=name)

        self.objectives = objectives

        if weights is None:
            weights = np.fill(1 / len(self.objectives), len(self.objectives))
        self.weights = np.asarray(weights) / sum(weights)

        self.epsilon = epsilon


    def fitness(self, indiv: Individual, adjusted=True) -> float:
        fitness_total = float("inf")
        fitness_values = [obj_func.fitness(indiv, adjusted) for obj_func in self.objectives]
        fitness_sum = sum(fitness_values)

        for fit, w in zip(fitness_values, self.weights):
            fitness_chevy = w * fit + self.epsilon * fitness_sum
            if fitness_total > fitness_chevy:
                fitness_total = fitness_chevy

        return fitness_total

    def objective(self, solution: Any) -> float:
        obj_total = float("inf")
        obj_values = [obj_func.objective(indiv) for obj_func in self.objectives]
        obj_sum = sum(obj_values)

        for obj, w in zip(obj_values, self.weights):
            obj_chevy = w * obj + self.epsilon * obj_sum
            if obj_total > obj_chevy:
                obj_total = obj_chevy

        return obj_total

    def repair_solution(self, solution: Any) -> Any:
        return solution

