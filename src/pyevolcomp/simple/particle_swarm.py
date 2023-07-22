from __future__ import annotations
from ..Initializers import UniformVectorInitializer
from ..Encodings import TypeCastEncoding
from ..Algorithms import PSO
from ..SearchMethods import GeneralSearch

def particle_swarm(objfunc: ObjectiveVectorFunc, params: dict) -> Search:
    """
    Instantiates a particle swarm algorithm to optimize the given objective function.

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

    encoding_str = params["encoding"] if "encoding" in params else "real"

    if encoding_str.lower() == "real":
        alg = _particle_swarm_real_vec(objfunc, params)
    elif encoding_str.lower() == "int":
        alg = _particle_swarm_int_vec(objfunc, params)
    elif encoding_str.lower() == "bin":
        alg = _particle_swarm_bin_vec(objfunc, params)
    else:
        raise ValueError(f"The encoding \"{encoding_str}\" does not exist, try \"real\", \"int\" or \"bin\"")
    
    return alg


def _particle_swarm_real_vec(objfunc, params):
    """
    Instantiates a particle swarm algorithm to optimize the given objective function.
    This objective function should accept real coded vectors.
    """

    pop_size = params["pop_size"] if "pop_size" in params else 100
    w = params["w"] if "w" in params else 0.7
    c1 = params["c1"] if "c1" in params else 1.5
    c2 = params["c2"] if "c2" in params else 1.5

    pop_initializer = UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size, dtype=float)
    
    search_strat = PSO(pop_initializer, {"w":w, "c1":c1, "c2":c2})

    return GeneralSearch(objfunc, search_strat, params=params)

def _particle_swarm_int_vec(objfunc, params):
    """
    Instantiates a particle swarm algorithm to optimize the given objective function.
    This objective function should accept real coded vectors.
    """

    pop_size = params["pop_size"] if "pop_size" in params else 100
    w = params["w"] if "w" in params else 0.7
    c1 = params["c1"] if "c1" in params else 1.5
    c2 = params["c2"] if "c2" in params else 1.5

    encoding = TypeCastEncoding(float, int)

    pop_initializer = UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size, dtype=float, encoding=encoding)
    
    search_strat = PSO(pop_initializer, {"w":w, "c1":c1, "c2":c2})

    return GeneralSearch(objfunc, search_strat, params=params)


def _particle_swarm_bin_vec(objfunc, params):
    """
    Instantiates a particle swarm algorithm to optimize the given objective function.
    This objective function should accept real coded vectors.
    """

    pop_size = params["pop_size"] if "pop_size" in params else 100
    w = params["w"] if "w" in params else 0.7
    c1 = params["c1"] if "c1" in params else 1.5
    c2 = params["c2"] if "c2" in params else 1.5

    encoding = TypeCastEncoding(float, bool)

    pop_initializer = UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size, dtype=float, encoding=encoding)
    
    search_strat = PSO(pop_initializer, {"w":w, "c1":c1, "c2":c2})

    return GeneralSearch(objfunc, search_strat, params=params)
