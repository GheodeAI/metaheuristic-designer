from ..Initializers import UniformVectorInitializer
from ..Algorithms import RandomSearch
from ..SearchMethods import GeneralSearch

def random_search(objfunc, params):
    encoding_str = params["encoding"] if "encoding" in params else "bin"

    if encoding_str.lower() == "bin":
        alg = random_search_bin_vec(objfunc, params)
    elif encoding_str.lower() == "int":
        alg = random_search_int_vec(objfunc, params)
    elif encoding_str.lower() == "real":
        alg = random_search_real_vec(objfunc, params)
    else:
        raise ValueError(f"The encoding \"{encoding_str}\" does not exist, try \"real\", \"int\" or \"bin\"")
    
    return alg

def random_search_bin_vec(objfunc, params):
    """
    Instantiates a genetic algorithm to optimize the given objective function.
    This objective function should accept binary coded vectors.
    """

    pop_initializer = UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size, dtype=bool)

    search_strat = RandomSearch(pop_initializer)

    return GeneralSearch(objfunc, search_strat, params=params)


def random_search_int_vec(objfunc, params):
    """
    Instantiates a genetic algorithm to optimize the given objective function.
    This objective function should accept integer coded vectors.
    """

    pop_initializer = UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size, dtype=int)

    search_strat = RandomSearch(pop_initializer)

    return GeneralSearch(objfunc, search_strat, params=params)


def random_search_real_vec(objfunc, params):
    """
    Instantiates a genetic algorithm to optimize the given objective function.
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
