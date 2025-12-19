from __future__ import annotations
from abc import abstractmethod
import numpy as np
from numpy import ndarray
from typing import Iterable
from ..objective_function import *


class MultiobjectiveFunc(ObjectiveFunc):
    def __init__(
        self,
        n_objectives: int,
        constraint_handler: ConstraintHandler = None,
        modes: Iterable[str] = "max",
        name: str = "Weighted average of functions",
        vectorized: bool = False,
        recalculate: bool = False,
    ):
        super().__init__(
            constraint_handler=constraint_handler,
            name=name,
            vectorized=vectorized,
            recalculate=recalculate
        )

        self.n_objectives = n_objectives
        self.modes = modes
        self.factors = np.ones(n_objectives)
        if modes is not None:
            if isinstance(modes, str) and modes == "min":
                self.factors *= -1
            else:
                self.factors[[i != "max" for i in modes]] = -1

    def fitness(self, population: Population, adjusted: bool = True) -> ndarray:
        fitness = population.fitness
        solutions = population.decode()
        if self.vectorized:
            if self.recalculate:
                solutions = solutions[population.fitness_calculated == 0, :]

            fitness_new = self.objective(solutions)
            if adjusted:
                fitness_new = self.factors * (fitness_new - self.constraint_handler.penalty(solutions))

            if self.recalculate:
                fitness[population.fitness_calculated == 0] = fitness_new
            else:
                fitness = fitness_new

        else:
            for idx, (solution, already_calculated) in enumerate(zip(solutions, population.fitness_calculated)):
                if self.recalculate or not already_calculated:
                    value = self.objective(solution)

                    if adjusted:
                        value = self.factors * (value - self.constraint_handler.penalty(solution))

                    fitness[idx] = value

        if self.recalculate:
            self.counter += int(population.pop_size)
        else:
            self.counter += int(population.pop_size - population.fitness_calculated.sum())

        population.fitness_calculated = np.ones_like(population.fitness_calculated)

        return fitness

    @abstractmethod
    def objective(self, solution: Any) -> ndarray:
        """"""


class MultiobjectiveVectorFunc(MultiobjectiveFunc):
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