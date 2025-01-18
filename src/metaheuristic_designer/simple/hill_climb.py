from __future__ import annotations
from ..ObjectiveFunc import ObjectiveVectorFunc
from ..Algorithm import Algorithm
from ..initializers import UniformVectorInitializer
from ..operators import OperatorVector
from ..encodings import TypeCastEncoding
from ..strategies import HillClimb
from ..algorithms import GeneralAlgorithm


def hill_climb(params: dict, objfunc: ObjectiveVectorFunc = None) -> Algorithm:
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
    algorithm: Algorithm
        Configured optimization algorithm.
    """

    if "encoding" not in params:
        raise ValueError('You must specify the encoding in the params structure, the options are "real", "int" and "bin"')

    encoding_str = params["encoding"]

    if encoding_str.lower() == "bin":
        alg = _hill_climb_bin_vec(params, objfunc)
    elif encoding_str.lower() == "int":
        alg = _hill_climb_int_vec(params, objfunc)
    elif encoding_str.lower() == "real":
        alg = _hill_climb_real_vec(params, objfunc)
    else:
        raise ValueError(f'The encoding "{encoding_str}" does not exist, try "real", "int" or "bin"')

    return alg


def _hill_climb_bin_vec(params, objfunc):
    """
    Instantiates a hill climbing algorithm to optimize the given objective function.
    This objective function should accept binary coded vectors.
    """

    mutstr = params.get("mutstr", 1)
    if objfunc is None:
        vecsize = params["vecsize"]
    else:
        vecsize = objfunc.vecsize

    encoding = TypeCastEncoding(int, bool)

    pop_initializer = UniformVectorInitializer(vecsize, 0, 1, pop_size=1, dtype=int, encoding=encoding)

    mutation_op = OperatorVector("Flip", {"N": mutstr})

    search_strat = HillClimb(pop_initializer, mutation_op)

    return GeneralAlgorithm(objfunc, search_strat, params=params)


def _hill_climb_int_vec(params, objfunc):
    """
    Instantiates a hill climbing algorithm to optimize the given objective function.
    This objective function should accept integer coded vectors.
    """

    mutstr = params.get("mutstr", 1)
    if objfunc is None:
        vecsize = params["vecsize"]
    else:
        vecsize = objfunc.vecsize
    min_val = params.get("min", objfunc.low_lim if objfunc else 0)
    max_val = params.get("max", objfunc.up_lim if objfunc else 100)

    pop_initializer = UniformVectorInitializer(vecsize, min_val, max_val, pop_size=1, dtype=int)

    mutation_op = OperatorVector(
        "MutRand",
        {
            "distrib": "Uniform",
            "Low": objfunc.low_lim,
            "Up": objfunc.up_lim,
            "N": mutstr,
        },
    )

    search_strat = HillClimb(pop_initializer, mutation_op)

    return GeneralAlgorithm(objfunc, search_strat, params=params)


def _hill_climb_real_vec(params, objfunc):
    """
    Instantiates a hill climbing algorithm to optimize the given objective function.
    This objective function should accept real coded vectors.
    """

    mutstr = params.get("mutstr", 1e-5)
    if objfunc is None:
        vecsize = params["vecsize"]
    else:
        vecsize = objfunc.vecsize
    min_val = params.get("min", objfunc.low_lim if objfunc else 0)
    max_val = params.get("max", objfunc.up_lim if objfunc else 100)

    pop_initializer = UniformVectorInitializer(vecsize, min_val, max_val, pop_size=1, dtype=float)

    mutation_op = OperatorVector("RandNoise", {"distrib": "Gauss", "F": mutstr})

    search_strat = HillClimb(pop_initializer, mutation_op)

    return GeneralAlgorithm(objfunc, search_strat, params=params)
