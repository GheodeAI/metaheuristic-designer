"""
Ready-to-run Random Search wrappers.
"""

from __future__ import annotations
from typing import Optional
import numpy as np

from metaheuristic_designer.encoding import Encoding
from metaheuristic_designer.objective_function import ObjectiveFunc
from ..initializers import UniformInitializer, PermInitializer
from ..encodings import TypeCastEncoding
from ..strategies import RandomSearch
from ..algorithms import Algorithm
from ..utils import RNGLike, check_random_state


def random_search_binary(objfunc: ObjectiveFunc, encoding: Optional[Encoding] = None, random_state: Optional[RNGLike] = None, **kwargs) -> Algorithm:
    """Random Search for binary-coded vectors.

    Parameters
    ----------
    objfunc : ObjectiveFunc
        The objective function to optimise.
    encoding : Encoding, optional
        Encoding; defaults to :class:`TypeCastEncoding` (int → bool).
    random_state : RNGLike, optional
        Random seed or generator.
    **kwargs
        Forwarded to :class:`Algorithm`.
    """

    random_state = check_random_state(random_state)
    encoding = TypeCastEncoding(int, bool) if encoding is None else encoding
    pop_initializer = UniformInitializer(objfunc.dimension, 0, 1, population_size=1, dtype=np.uint8, encoding=encoding, random_state=random_state)
    search_strat = RandomSearch(pop_initializer, random_state=random_state)
    return Algorithm(objfunc, search_strat, **kwargs)


def random_search_permutation(
    objfunc: ObjectiveFunc, encoding: Optional[Encoding] = None, random_state: Optional[RNGLike] = None, **kwargs
) -> Algorithm:
    """Random Search for permutation-coded vectors.

    Parameters
    ----------
    objfunc : ObjectiveFunc
        The objective function to optimise.
    encoding : Encoding, optional
        Encoding applied to the genotype.
    random_state : RNGLike, optional
        Random seed or generator.
    **kwargs
        Forwarded to :class:`Algorithm`.
    """

    random_state = check_random_state(random_state)
    pop_initializer = PermInitializer(objfunc.dimension, population_size=1, encoding=encoding, random_state=random_state)
    search_strat = RandomSearch(pop_initializer, random_state=random_state)
    return Algorithm(objfunc, search_strat, **kwargs)


def random_search_discrete(
    objfunc: ObjectiveFunc, encoding: Optional[Encoding] = None, random_state: Optional[RNGLike] = None, **kwargs
) -> Algorithm:
    """Random Search for integer-coded vectors.

    Parameters
    ----------
    objfunc : ObjectiveFunc
        The objective function to optimise.
    encoding : Encoding, optional
        Encoding applied to the genotype.
    random_state : RNGLike, optional
        Random seed or generator.
    **kwargs
        Forwarded to :class:`Algorithm`.
    """

    random_state = check_random_state(random_state)
    pop_initializer = UniformInitializer(
        objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=1, dtype=int, encoding=encoding, random_state=random_state
    )
    search_strat = RandomSearch(pop_initializer, random_state=random_state)
    return Algorithm(objfunc, search_strat, **kwargs)


def random_search_real(objfunc: ObjectiveFunc, encoding: Optional[Encoding] = None, random_state: Optional[RNGLike] = None, **kwargs) -> Algorithm:
    """Random Search for real-coded vectors.

    Parameters
    ----------
    objfunc : ObjectiveFunc
        The objective function to optimise.
    encoding : Encoding, optional
        Encoding applied to the genotype.
    random_state : RNGLike, optional
        Random seed or generator.
    **kwargs
        Forwarded to :class:`Algorithm`.
    """

    random_state = check_random_state(random_state)
    pop_initializer = UniformInitializer(
        objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=1, dtype=float, encoding=encoding, random_state=random_state
    )
    search_strat = RandomSearch(pop_initializer, random_state=random_state)
    return Algorithm(objfunc, search_strat, **kwargs)
