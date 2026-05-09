from __future__ import annotations
import numpy as np
from ..initializers import UniformInitializer, PermInitializer
from ..encodings import TypeCastEncoding
from ..strategies import RandomSearch
from ..algorithms import Algorithm
from ..utils import check_random_state


def random_search_binary(objfunc, encoding=None, random_state=None, **kwargs):
    """
    Instantiates a hill climbing algorithm to optimize the given objective function.
    This objective function should accept binary coded vectors.
    """

    random_state = check_random_state(random_state)
    encoding = TypeCastEncoding(int, bool) if encoding is None else encoding
    pop_initializer = UniformInitializer(objfunc.dimension, 0, 1, population_size=1, dtype=np.uint8, encoding=encoding, random_state=random_state)
    search_strat = RandomSearch(pop_initializer, random_state=random_state)
    return Algorithm(objfunc, search_strat, **kwargs)


def random_search_permutation(objfunc, encoding=None, random_state=None, **kwargs):
    """
    Instantiates a hill climbing algorithm to optimize the given objective function.
    This objective function should accept integer coded vectors.
    """

    random_state = check_random_state(random_state)
    pop_initializer = PermInitializer(objfunc.dimension, population_size=1, encoding=encoding, random_state=random_state)
    search_strat = RandomSearch(pop_initializer, random_state=random_state)
    return Algorithm(objfunc, search_strat, **kwargs)


def random_search_discrete(objfunc, encoding=None, random_state=None, **kwargs):
    """
    Instantiates a hill climbing algorithm to optimize the given objective function.
    This objective function should accept integer coded vectors.
    """

    random_state = check_random_state(random_state)
    pop_initializer = UniformInitializer(
        objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=1, dtype=int, encoding=encoding, random_state=random_state
    )
    search_strat = RandomSearch(pop_initializer, random_state=random_state)
    return Algorithm(objfunc, search_strat, **kwargs)


def random_search_real(objfunc, encoding=None, random_state=None, **kwargs):
    """
    Instantiates a hill climbing algorithm to optimize the given objective function.
    This objective function should accept real coded vectors.
    """

    random_state = check_random_state(random_state)
    pop_initializer = UniformInitializer(
        objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=1, dtype=float, encoding=encoding, random_state=random_state
    )
    search_strat = RandomSearch(pop_initializer, random_state=random_state)
    return Algorithm(objfunc, search_strat, **kwargs)
