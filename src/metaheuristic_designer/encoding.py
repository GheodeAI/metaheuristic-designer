"""
Base class for the Encoding module.

This module implements a way to have a different representation in the inner working
of the algorithm and the result of the procedure.
"""

from __future__ import annotations
from typing import Iterable, Any, Callable
from abc import ABC, abstractmethod
import numpy as np
from numpy import ndarray
from .parametrizable_mixin import ParametrizableMixin
from .utils import MatrixLike


class Encoding(ParametrizableMixin, ABC):
    """
    Abstract Encoding class

    This class transforms between phenotype and genotype.
    """

    def __init__(self, decode_as_array: bool = False, **kwargs):
        super().__init__()
        self.decode_as_array = decode_as_array
        self.store_kwargs(**kwargs)

    @abstractmethod
    def encode_func(self, solution: Iterable, **kwargs) -> MatrixLike:
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
    def decode_func(self, population_matrix: MatrixLike) -> Iterable:
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

    def encode(self, solutions: Iterable) -> MatrixLike:
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

        return self.encode_func(solutions)

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

        solutions = self.decode_func(population)

        if self.decode_as_array:
            solutions = np.asarray(solutions)

        return solutions

    def get_state(self) -> dict:
        return {}


class DefaultEncoding(Encoding):
    """
    Default encoder that uses the genotype directly as a solution.
    """

    def __init__(self, decode_as_array: bool = True):
        super().__init__(decode_as_array=decode_as_array)

    def encode_func(self, solution: Iterable) -> MatrixLike:
        return solution

    def decode_func(self, population: MatrixLike) -> Iterable:
        return population


class EncodingFromLambda(Encoding):
    """
    Decoder that uses user specified functions.
    """

    def __init__(self, encode_fn: Callable, decode_fn: Callable):
        super().__init__()
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn

    def encode_func(self, solution: Iterable) -> MatrixLike:
        return self.encode_fn(solution)

    def decode_func(self, population: MatrixLike) -> Iterable:
        return self.decode_fn(population)
