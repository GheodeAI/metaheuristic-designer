from __future__ import annotations
from ..ObjectiveFunc import ObjectiveVectorFunc
from ..Algorithm import Algorithm
from ..initializers import UniformVectorInitializer
from ..encodings import TypeCastEncoding
from ..strategies import RandomSearch
from ..algorithms import GeneralAlgorithm


def random_search(params: dict, objfunc: ObjectiveVectorFunc = None) -> Algorithm:
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
        raise ValueError(f'You must specify the encoding in the params structure, the options are "real", "int" and "bin"')

    encoding_str = params["encoding"]

    if encoding_str.lower() == "bin":
        alg = _random_search_bin_vec(params, objfunc)
    elif encoding_str.lower() == "int":
        alg = _random_search_int_vec(params, objfunc)
    elif encoding_str.lower() == "real":
        alg = _random_search_real_vec(params, objfunc)
    else:
        raise ValueError(f'The encoding "{encoding_str}" does not exist, try "real", "int" or "bin"')

    return alg


def _random_search_bin_vec(params, objfunc):
    """
    Instantiates a random search algorithm to optimize the given objective function.
    This objective function should accept binary coded vectors.
    """

    pop_size = params.get("pop_size", 100)

    if objfunc is None:
        vecsize = params["vecsize"]
    else:
        vecsize = objfunc.vecsize

    encoding = TypeCastEncoding(int, bool)

    pop_initializer = UniformVectorInitializer(vecsize, 0, 1, pop_size=pop_size, dtype=int, encoding=encoding)

    search_strat = RandomSearch(pop_initializer)

    return GeneralAlgorithm(objfunc, search_strat, params=params)


def _random_search_int_vec(params, objfunc):
    """
    Instantiates a random search algorithm to optimize the given objective function.
    This objective function should accept integer coded vectors.
    """

    pop_size = params.get("pop_size", 100)

    if objfunc is None:
        vecsize = params["vecsize"]
    else:
        vecsize = objfunc.vecsize
    min_val = params.get("min", objfunc.low_lim if objfunc else 0)
    max_val = params.get("max", objfunc.up_lim if objfunc else 100)

    pop_initializer = UniformVectorInitializer(vecsize, min_val, max_val, pop_size=pop_size, dtype=int)

    search_strat = RandomSearch(pop_initializer)

    return GeneralAlgorithm(objfunc, search_strat, params=params)


def _random_search_real_vec(params, objfunc):
    """
    Instantiates a random search algorithm to optimize the given objective function.
    This objective function should accept real coded vectors.
    """

    pop_size = params.get("pop_size", 100)

    if objfunc is None:
        vecsize = params["vecsize"]
    else:
        vecsize = objfunc.vecsize
    min_val = params.get("min", objfunc.low_lim if objfunc else 0)
    max_val = params.get("max", objfunc.up_lim if objfunc else 100)

    pop_initializer = UniformVectorInitializer(vecsize, min_val, max_val, pop_size=pop_size, dtype=float)

    search_strat = RandomSearch(pop_initializer)

    return GeneralAlgorithm(objfunc, search_strat, params=params)
