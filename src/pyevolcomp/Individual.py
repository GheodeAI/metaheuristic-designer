from __future__ import annotations
from copy import copy
import numpy as np


class Individual:
    """
    Individual that holds a tentative solution with its fitness.
    """

    def __init__(self, objfunc: ObjectiveFunc, vector: np.ndarray, speed: np.ndarray = None, operator: Operator = None):
        """
        Constructor of the Individual class.
        """

        self.objfunc = objfunc
        self._genotype = vector
        self.speed = speed
        if speed is None and isinstance(vector, np.ndarray):
            self.speed = np.zeros_like(vector)
        self.operator = operator
        self._fitness = 0
        self.fitness_calculated = False
        self.best = vector
        self.is_dead = False

    def __copy__(self) -> Individual:
        """
        Returns a copy of the Individual.
        """

        copied_ind = Individual(self.objfunc, copy(self._genotype), copy(self.speed), self.operator)
        copied_ind._fitness = self._fitness
        copied_ind.fitness_calculated = self.fitness_calculated
        copied_ind.best = copy(self.best)
        return copied_ind

    @property
    def genotype(self) -> np.ndarray:
        """
        Gets the value of the vector.
        """

        return self._genotype

    @genotype.setter
    def genotype(self, vector: np.ndarray):
        """
        Sets the value of the vector.
        """

        self.fitness_calculated = False
        self._genotype = vector

    def store_best(self, past_indiv: Individual):
        """
        Stores the vector that yeided the best fitness between the one the indiviudal has and another input vector
        """

        if self.fitness < past_indiv.fitness:
            self.best = past_indiv.genotype

    def reproduce(self, population: List[Individual]) -> Individual:
        """
        Apply the operator to obtain a new individual.
        """

        new_indiv = self.operator(self, population, self.objfunc)
        new_indiv.genotype = self.objfunc.repair_solution(new_indiv.genotype)
        return Individual(self.objfunc, new_vector, self.speed, self.operator)

    def apply_speed(self) -> Individual:
        """
        Apply the speed to obtain an individual with a new position.
        """

        return Individual(self.objfunc, self._genotype + self.speed, self.speed, self.operator)

    @property
    def fitness(self) -> float:
        """
        Obtain the fitness of the individual, optimized to be calculated only once per individual.
        """

        if not self.fitness_calculated:
            self.fitness = self.objfunc(self)
        return self._fitness

    @fitness.setter
    def fitness(self, fit: float):
        """
        Obtain the fitness of the individual, optimized to be calculated only once per individual.
        """

        self._fitness = fit
        self.fitness_calculated = True
