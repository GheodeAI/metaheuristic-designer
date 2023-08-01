from __future__ import annotations
from ..Initializers import UniformVectorInitializer
from ..Encodings import TypeCastEncoding
from ..Algorithms import RandomSearch
from ..SearchMethods import GeneralSearch

def random_search(objfunc: ObjectiveVectorFunc, params: dict) -> Search:
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
    algorithm: Search
        Configured optimization algorithm.
    """

    encoding_str = params["encoding"] if "encoding" in params else "bin"

    if encoding_str.lower() == "bin":
        alg = _random_search_bin_vec(objfunc, params)
    elif encoding_str.lower() == "int":
        alg = _random_search_int_vec(objfunc, params)
    elif encoding_str.lower() == "real":
        alg = _random_search_real_vec(objfunc, params)
    else:
        raise ValueError(f"The encoding \"{encoding_str}\" does not exist, try \"real\", \"int\" or \"bin\"")
    
    return alg

def _random_search_bin_vec(objfunc, params):
    """
    Instantiates a random search algorithm to optimize the given objective function.
    This objective function should accept binary coded vectors.
    """

    encoding = TypeCastEncoding(int, bool)

    pop_initializer = UniformVectorInitializer(objfunc.vecsize, 0, 1, pop_size=1, dtype=int, encoding=encoding)

    search_strat = RandomSearch(pop_initializer)

    return GeneralSearch(objfunc, search_strat, params=params)


def _random_search_int_vec(objfunc, params):
    """
    Instantiates a random search algorithm to optimize the given objective function.
    This objective function should accept integer coded vectors.
    """

    pop_initializer = UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size, dtype=int)

    search_strat = RandomSearch(pop_initializer)

    return GeneralSearch(objfunc, search_strat, params=params)


def _random_search_real_vec(objfunc, params):
    """
    Instantiates a random search algorithm to optimize the given objective function.
    This objective function should accept real coded vectors.
    """

    pop_size = params["pop_size"] if "pop_size" in params else 100
    n_parents = params["n_parents"] if "n_parents" in params else 20
    pcross = params["pcross"] if "pcross" in params else 0.8
    pmut = params["pmut"] if "pmut" in params else 0.1
    mutstr = params["mutstr"] if "mutstr" in params else 1e-5 

    pop_initializer = UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size, dtype=float)
    
    search_strat = RandomSearch(pop_initializer)

    return GeneralSearch(objfunc, search_strat, params=params)
