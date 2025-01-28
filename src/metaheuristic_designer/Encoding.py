from __future__ import annotations
from typing import Iterable, Any
from abc import ABC, abstractmethod
import warnings
from numpy import ndarray


class Encoding(ABC):
    """
    Abstract Encoding class

    This class transforms between phenotype and genotype.
    """

    def __init__(self, vectorized=False):
        self.vectorized = vectorized

    @abstractmethod
    def encode_func(self, solution: Any) -> ndarray:
        """ """

    @abstractmethod
    def decode_func(self, indiv: ndarray) -> Any:
        """ """

    def encode(self, solutions: Iterable) -> ndarray:
        """
        Encodes a viable solution to our problem to the encoding used in each individual of the algorithm.

        Parameters
        ----------
        phenotype: Any
            Information that should be encoded.

        Returns
        -------
        genotype: Any
            Encoded information of the phenotype.
        """

        population = None
        if self.vectorized:
            population = self.encode_func(solutions)
        else:
            population = np.asarray([self.encode_single(indiv) for indiv in solutions])

        return population

    def decode(self, population: ndarray) -> Iterable:
        """
        Decodes the contents of an individual to a viable solution to our problem.

        Parameters
        ----------
        genotype: Any
            Information that should be decoded.

        Returns
        -------
        phenotype: Any
            Decoded information of the genotype.
        """

        solutions = None
        if self.vectorized:
            solutions = self.decode_func(population)
        else:
            solutions = [self.decode_single(indiv) for indiv in population]

        return solutions
