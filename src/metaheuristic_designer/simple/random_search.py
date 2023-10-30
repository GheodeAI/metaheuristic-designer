from __future__ import annotations
from ..initializers import UniformVectorInitializer
from ..encodings import TypeCastEncoding
from ..strategies import RandomSearch
from ..algorithms import GeneralAlgorithm


def random_search(objfunc: ObjectiveVectorFunc, params: dict) -> Algorithm:
    """
    Instantiates a random search algorithm to optimize the given objective function.

    Parameters
    ----------
    objfunc: ObjectiveFunc
        Objective function to be optimized.
    params: ParamScheduler or dict, optional
        Dictionary of parameters of the algorithm.

    Returns
    -------
    algorithm: Algorithm
        Configured optimization algorithm.
    """

    if "encoding" not in params:
        raise ValueError(
            f'You must specify the encoding in the params structure, the options are "real", "int" and "bin"'
        )

    encoding_str = params["encoding"]

    if encoding_str.lower() == "bin":
        alg = _random_search_bin_vec(objfunc, params)
    elif encoding_str.lower() == "int":
        alg = _random_search_int_vec(objfunc, params)
    elif encoding_str.lower() == "real":
        alg = _random_search_real_vec(objfunc, params)
    else:
        raise ValueError(
            f'The encoding "{encoding_str}" does not exist, try "real", "int" or "bin"'
        )

    return alg


def _random_search_bin_vec(objfunc, params):
    """
    Instantiates a random search algorithm to optimize the given objective function.
    This objective function should accept binary coded vectors.
    """

    encoding = TypeCastEncoding(int, bool)

    pop_initializer = UniformVectorInitializer(
        objfunc.vecsize, 0, 1, pop_size=1, dtype=int, encoding=encoding
    )

    search_strat = RandomSearch(pop_initializer)

    return GeneralAlgorithm(objfunc, search_strat, params=params)


def _random_search_int_vec(objfunc, params):
    """
    Instantiates a random search algorithm to optimize the given objective function.
    This objective function should accept integer coded vectors.
    """

    pop_initializer = UniformVectorInitializer(
        objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size, dtype=int
    )

    search_strat = RandomSearch(pop_initializer)

    return GeneralAlgorithm(objfunc, search_strat, params=params)


def _random_search_real_vec(objfunc, params):
    """
    Instantiates a random search algorithm to optimize the given objective function.
    This objective function should accept real coded vectors.
    """

    pop_size = params.get("pop_size", 100)
    n_parents = params.get("n_parents", 20)
    pcross = params.get("pcross", 0.8)
    pmut = params.get("pmut", 0.1)
    mutstr = params.get("mutstr", 1e-5)

    pop_initializer = UniformVectorInitializer(
        objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size, dtype=float
    )

    search_strat = RandomSearch(pop_initializer)

    return GeneralAlgorithm(objfunc, search_strat, params=params)
