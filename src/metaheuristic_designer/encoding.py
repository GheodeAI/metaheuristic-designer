"""
Base class for the Encoding module.

This module implements a way to have a different representation in the inner working
of the algorithm and the result of the procedure.
"""

from __future__ import annotations
from typing import Iterable, Callable
from abc import ABC, abstractmethod
from .parametrizable_mixin import ParametrizableMixin
from .utils import MatrixLike


class Encoding(ParametrizableMixin, ABC):
    """Translate between internal genotypes and problem-specific phenotypes.

    An :class:`Encoding` is responsible for converting a population
    matrix (the internal representation used by operators) into a
    collection of solutions that the objective function understands,
    and vice versa.

    Parameters
    ----------
    decode_as_array : bool, optional
        If ``True``, :meth:`decode` returns a NumPy array instead of
        an iterable of arbitrary objects. Default ``False``.
    name : str, optional
        Display name for this encoding.
    \\*\\*kwargs
        Additional keyword arguments stored as schedulable
        parameters.
    """

    def __init__(self, decode_as_array: bool = False, name=None, **kwargs):
        super().__init__()
        self.name = name
        self.decode_as_array = decode_as_array
        self.store_kwargs(**kwargs)

    def gather_params(self) -> dict:
        """
        Overridable thin wrapper around get_params
        """

        return self.get_params()

    @abstractmethod
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

    @abstractmethod
    def decode(self, population: MatrixLike) -> Iterable:
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

    def get_state(self) -> dict:
        return {}


class DefaultEncoding(Encoding):
    """Identity encoding - the internal genotype is used directly.

    No transformation is applied; :meth:`encode` and :meth:`decode`
    return their arguments unchanged.  This is the encoding used
    when no other is specified.

    Parameters
    ----------
    decode_as_array : bool, optional
        See :class:`Encoding`. Default ``True``.
    """

    def __init__(self, decode_as_array: bool = True):
        super().__init__(decode_as_array=decode_as_array)

    def encode(self, solution: Iterable) -> MatrixLike:
        return solution

    def decode(self, population: MatrixLike) -> Iterable:
        return population


class EncodingFromLambda(Encoding):
    """Encoding built from two callables.

    Parameters
    ----------
    encode_fn : callable
        ``(solutions) -> population_matrix``.
    decode_fn : callable
        ``(population_matrix) -> solutions``.
    decode_as_array : bool, optional
        See :class:`Encoding`.
    \\*\\*kwargs
        Forwarded to :class:`Encoding`.
    """

    def __init__(self, encode_fn: Callable, decode_fn: Callable, **kwargs):
        super().__init__(**kwargs)
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn

    def encode(self, solution: Iterable) -> MatrixLike:
        return self.encode_fn(solution)

    def decode(self, population: MatrixLike) -> Iterable:
        return self.decode_fn(population)
