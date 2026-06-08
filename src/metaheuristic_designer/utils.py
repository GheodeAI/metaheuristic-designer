"""Utility functions, type aliases, and a JSON encoder used across the library."""

from typing import Optional
import json
import numbers
from enum import Enum
import numpy as np

null_aliases = {"null", "nothing", "identity", "passthrough"}

RNGLike = int | np.random.Generator

RealVector = np.ndarray[tuple[int], np.floating]
RealMatrix = np.ndarray[tuple[int, int], np.floating]
RealTensor = np.ndarray[tuple[int, ...], np.floating]

IntVector = np.ndarray[tuple[int], np.integer]
IntMatrix = np.ndarray[tuple[int, int], np.integer]
IntTensor = np.ndarray[tuple[int, ...], np.integer]

BinVector = np.ndarray[tuple[int], np.uint8 | np.bool]
BinMatrix = np.ndarray[tuple[int, int], np.uint8 | np.bool]
BinTensor = np.ndarray[tuple[int, ...], np.uint8 | np.bool]

ScalarLike = np.number | float | int
VectorLike = RealVector | IntVector | BinVector
MatrixLike = RealMatrix | IntMatrix | BinMatrix
TensorLike = RealTensor | IntTensor | BinTensor

MaskLike = IntTensor | BinTensor


class TerminationException(Exception):
    """
    Custom exception to handle SIGTERM
    """


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that can serialize NumPy scalars, arrays, and Enums.

    Use this encoder with ``json.dumps`` or ``json.dump`` when your
    data structure contains NumPy integers, floats, or arrays, or
    when you need to serialize Enum values as their string names.
    """

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, Enum):
            return str(o)
        return super().default(o)


def check_rng(seed: Optional[RNGLike]) -> np.random.Generator:
    """Turn seed into an np.random.Generator instance.

    Original implementation adapted from:
    https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/validation.py
    BSD 3-Clause License, Copyright (c) 2007-2025 The scikit-learn developers.

    Parameters
    ----------
    seed : None, int or instance of Generator
        If seed is None, return the Generator singleton used by np.random.
        If seed is an int, return a new Generator instance seeded with seed.
        If seed is already a Generator instance, return it.
        Otherwise raise ValueError.

    Returns
    -------
    :class:`numpy:numpy.random.Generator`
        The random state object based on `seed` parameter.

    """

    if seed is None or seed is np.random:
        return np.random.default_rng()
    if isinstance(seed, numbers.Integral):
        return np.random.default_rng(seed)
    if isinstance(seed, np.random.Generator):
        return seed
    raise ValueError("%r cannot be used to seed a numpy.random.Generator instance" % seed)


def per_individual(func):
    """Decorator that applies a function to each row of a 2-D array.

    The wrapped function receives a single row (a 1-D array) and
    any keyword arguments, and must return a 1-D array of the same
    length.  The decorator loops over rows and stacks the results
    back into a 2-D array.

    Parameters
    ----------
    func : callable
        A function ``(row, **kwargs) -> 1-D array``.

    Returns
    -------
    callable
        A function that accepts a 2-D matrix and returns a 2-D array.
    """

    def wrapper(matrix, **kwargs):
        return np.array([func(row, **kwargs) for row in matrix])

    return wrapper


def per_individual_list(func):
    """Decorator that applies a function to each element of a list.

    The wrapped function receives a single element and any keyword
    arguments, and returns a transformed element.  The decorator
    loops over the list and returns a new list of the results.

    Parameters
    ----------
    func : callable
        A function ``(value, **kwargs) -> Any``.

    Returns
    -------
    callable
        A function that accepts a list and returns a list.
    """

    def wrapper(values, **kwargs):
        return [func(value, **kwargs) for value in values]

    return wrapper
