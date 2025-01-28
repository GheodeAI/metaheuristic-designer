from __future__ import annotations
import numpy as np
from typing import Iterable
from ..ObjectiveFunc import *


class ChebychevMultiObjectiveFunc(ObjectiveFunc):
    """
    https://doi.org/10.1007/
    """
    
    def __init__(
        self,
        objectives: Iterable[ObjectiveFunc],
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


    def fitness(self, population: Population, adjusted=True) -> ndarray:
        fitness_total = np.full(population.pop_size, np.inf)
        fitness_values = np.vstack([obj_func.fitness(population, adjusted) for obj_func in self.objectives]).T
        fitness_sum = np.sum(fitness_values, axis=1)

        for fit, w in zip(fitness_values, self.weights):
            fitness_chevy = w * fit + self.epsilon * fitness_sum
            # if fitness_total > fitness_chevy:
            #     fitness_total = fitness_chevy
            fitness_total[fitness_total > fitness_chevy] = fitness_chevy[fitness_total > fitness_chevy]

        return fitness_total

    def objective(self, solution: Any) -> ndarray:
        # obj_total = float("inf")
        # obj_values = [obj_func.objective(solution) for obj_func in self.objectives]
        # obj_sum = sum(obj_values)

        # for obj, w in zip(obj_values, self.weights):
        #     obj_chevy = w * obj + self.epsilon * obj_sum
        #     if obj_total > obj_chevy:
        #         obj_total = obj_chevy

        # return obj_total

        return np.array([obj_func.objective(solution) for obj_func in self.objectives])

    def repair_solution(self, solution: Any) -> Any:
        return solution

