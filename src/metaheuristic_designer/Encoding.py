from __future__ import annotations
from typing import Iterable, Any
from abc import ABC, abstractmethod
import warnings
import numpy as np
from numpy import ndarray


class Encoding(ABC):
    """
    Abstract Encoding class

    This class transforms between phenotype and genotype.
    """

    def __init__(self, vectorized=False, decode_as_array=False):
        self.vectorized = vectorized
        self.decode_as_array = decode_as_array

    @abstractmethod
    def encode_func(self, solution: Any) -> ndarray:
        """
        Convert a solution into an individual. (If vectorized is set it converts a list of solutions into a matrix)

        Parameters
        ----------
        solution: Any
            Solutions that should be encoded.

        Returns
        -------
        individual: ndarray
            Individual vector.
        """

    @abstractmethod
    def decode_func(self, indiv: ndarray) -> Any:
        """
        Convert an individual as a vector into an individual. (If vectorized is set it converts a list of solutions into a matrix)

        Parameters
        ----------
        solution: Any
            Solutions that should be encoded.

        Returns
        -------
        individual: ndarray
            Individual vector.
        """

    def encode(self, solutions: Iterable) -> ndarray:
        """
        Encodes a list of solutions to our problem to an population matrix.

        Parameters
        ----------
        solutions: Iterable
            Solutions that should be encoded.

        Returns
        -------
        population: ndarray
            Population array.
        """

        population = None
        if self.vectorized:
            population = self.encode_func(solutions)
        else:
            population = np.asarray([self.encode_func(indiv) for indiv in solutions])

        return population

    def decode(self, population: ndarray) -> Iterable:
        """
        Decodes a population matrix into a list/array of solutions.

        Parameters
        ----------
        population: ndarray
            Population that should be decoded.

        Returns
        -------
        solutions: Iterable
            List/array of solutions.
        """

        solutions = None
        if self.vectorized:
            solutions = self.decode_func(population)
        else:
            solutions = [self.decode_func(indiv) for indiv in population]

        if self.decode_as_array:
            solutions = np.asarray(solutions)

        return solutions
