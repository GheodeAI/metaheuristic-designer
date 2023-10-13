from __future__ import annotations
from copy import copy
import numpy as np
from numpy import ndarray
from .encodings import DefaultEncoding
from .utils import RAND_GEN


class Individual:
    """
    Individual that holds a tentative solution with its fitness.

    Parameters
    ----------
    objfunc: ObjectiveFunc
    genotype: Any
    speed: ndarray, optional
    encoding: Encoding, optional
    """

    _last_id = 0

    def __init__(
        self,
        objfunc: ObjectiveFunc,
        genotype: Any,
        speed: ndarray = None,
        encoding: Encoding = None,
    ):
        """
        Constructor of the Individual class.
        """

        self.id = Individual._last_id
        Individual._last_id += 1

        self.objfunc = objfunc
        self._genotype = genotype

        if speed is None and isinstance(genotype, np.ndarray):
            speed = RAND_GEN.random(size=genotype.shape)
        self.speed = speed

        self._fitness = 0
        self.best_fitness = None
        self.fitness_calculated = False
        self.best = genotype

        if encoding is None:
            encoding = DefaultEncoding()
        self.encoding = encoding

    def __copy__(self) -> Individual:
        """
        Returns a copy of the Individual.
        """

        copied_ind = Individual(
            self.objfunc, copy(self._genotype), copy(self.speed), self.encoding
        )
        copied_ind._fitness = self._fitness
        copied_ind.fitness_calculated = self.fitness_calculated
        copied_ind.best = copy(self.best)
        return copied_ind

    @property
    def genotype(self) -> ndarray:
        """
        The encoded information represented the individual.
        """

        return self._genotype

    @genotype.setter
    def genotype(self, vector: ndarray):
        """
        Sets the value of the vector.
        """

        self.fitness_calculated = False
        self._genotype = vector

    def apply_speed(self) -> Individual:
        """
        Apply the speed to obtain an individual with a new position.

        Returns
        -------
        modified_individual: Individual
            Individual with the speed applied.
        """

        return Individual(
            self.objfunc, self._genotype + self.speed, self.speed, self.encoding
        )

    @property
    def fitness(self) -> float:
        """
        The fitness of the individual, optimized to be calculated only once per individual.
        """

        if not self.fitness_calculated:
            self.fitness = self.objfunc(self)
        return self._fitness

    @fitness.setter
    def fitness(self, fit: float):
        """
        Manually sets a fitness to the individual.
        """

        if self.best_fitness is None or self.best_fitness < fit:
            self.best_fitness = fit
            self.best = self.genotype

        self._fitness = fit
        self.fitness_calculated = True

    def get_state(self, show_speed: bool = True, show_best: bool = False) -> dict:
        """
        Gets the current state of the algorithm as a dictionary.

        Parameters
        ----------
        show_speed: bool, optional
            Save the speed of the individual.
        show_best: bool, optional
            Save the best parent of this individual.

        Returns
        -------
        state: dict
            The current state of this individual.
        """

        data = {"genotype": self._genotype, "fitness": self._fitness}

        if show_speed:
            data["speed"] = self.speed

        if show_best:
            data["best_genotype"] = self.best

        return data
