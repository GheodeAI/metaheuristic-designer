from __future__ import annotations
from ..Initializers import UniformVectorInitializer
from ..Operators import OperatorInt, OperatorReal, OperatorBinary
from ..Algorithms import HillClimb
from ..Encodings import TypeCastEncoding
from ..SearchMethods import GeneralSearch

def hill_climb(objfunc: ObjectiveVectorFunc, params: dict) -> Search:
    """
    Instantiates a hill climbing algorithm to optimize the given objective function.

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

    encoding_str = params.get("encoding", "bin")

    if encoding_str.lower() == "bin":
        alg = _hill_climb_bin_vec(objfunc, params)
    elif encoding_str.lower() == "int":
        alg = _hill_climb_int_vec(objfunc, params)
    elif encoding_str.lower() == "real":
        alg = _hill_climb_real_vec(objfunc, params)
    else:
        raise ValueError(f"The encoding \"{encoding_str}\" does not exist, try \"real\", \"int\" or \"bin\"")
    
    return alg

def _hill_climb_bin_vec(objfunc, params):
    """
    Instantiates a hill climbing algorithm to optimize the given objective function.
    This objective function should accept binary coded vectors.
    """

    mutstr = params.get("mutstr", 1)

    encoding = TypeCastEncoding(int, bool)

    pop_initializer = UniformVectorInitializer(objfunc.vecsize, 0, 1, pop_size=1, dtype=int, encoding=encoding)

    mutation_op = OperatorBinary("Flip", {"N":mutstr})

    search_strat = HillClimb(pop_initializer, mutation_op)

    return GeneralSearch(objfunc, search_strat, params=params)


def _hill_climb_int_vec(objfunc, params):
    """
    Instantiates a hill climbing algorithm to optimize the given objective function.
    This objective function should accept integer coded vectors.
    """

    mutstr = params.get("mutstr", 1)

    pop_initializer = UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=1, dtype=int)

    mutation_op = OperatorInt("MutRand", {"method":"Uniform", "Low":objfunc.low_lim, "Up":objfunc.up_lim, "N":mutstr})

    search_strat = HillClimb(pop_initializer, mutation_op)

    return GeneralSearch(objfunc, search_strat, params=params)


def _hill_climb_real_vec(objfunc, params):
    """
    Instantiates a hill climbing algorithm to optimize the given objective function.
    This objective function should accept real coded vectors.
    """

    mutstr = params.get("mutstr", 1e-5 )

    pop_initializer = UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=1, dtype=float)

    mutation_op = OperatorReal("RandNoise", {"method":"Gauss", "F":mutstr})
    
    search_strat = HillClimb(pop_initializer, mutation_op)

    return GeneralSearch(objfunc, search_strat, params=params)