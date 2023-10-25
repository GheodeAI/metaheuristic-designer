from __future__ import annotations
from ..initializers import UniformVectorInitializer
from ..operators import OperatorInt, OperatorReal, OperatorBinary
from ..encodings import TypeCastEncoding
from ..strategies import SA
from ..algorithms import GeneralAlgorithm


def simulated_annealing(objfunc: ObjectiveVectorFunc, params: dict) -> Algorithm:
    """
    Instantiates a simulated annealing algorithm to optimize the given objective function.

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
        alg = _simulated_annealing_bin_vec(objfunc, params)
    elif encoding_str.lower() == "int":
        alg = _simulated_annealing_int_vec(objfunc, params)
    elif encoding_str.lower() == "real":
        alg = _simulated_annealing_real_vec(objfunc, params)
    else:
        raise ValueError(
            f'The encoding "{encoding_str}" does not exist, try "real", "int" or "bin"'
        )

    return alg


def _simulated_annealing_bin_vec(objfunc, params):
    """
    Instantiates a simulated annealing algorithm to optimize the given objective function.
    This objective function should accept binary coded vectors.
    """

    n_iter = params.get("iter", 100)
    temp_init = params.get("temp_init", 100)
    alpha = params.get("alpha", 0.99)
    mutstr = params.get("mutstr", 1)

    encoding = TypeCastEncoding(int, bool)

    pop_initializer = UniformVectorInitializer(
        objfunc.vecsize, 0, 1, pop_size=1, dtype=int, encoding=encoding
    )

    mutation_op = OperatorBinary("Flip", {"N": mutstr})

    search_strat = SA(
        pop_initializer,
        mutation_op,
        {"iter": n_iter, "temp_init": temp_init, "alpha": alpha},
    )

    return GeneralAlgorithm(objfunc, search_strat, params=params)


def _simulated_annealing_int_vec(objfunc, params):
    """
    Instantiates a simulated annealing algorithm to optimize the given objective function.
    This objective function should accept integer coded vectors.
    """

    n_iter = params.get("iter", 100)
    temp_init = params.get("temp_init", 100)
    alpha = params.get("alpha", 0.99)
    mutstr = params.get("mutstr", 1)

    pop_initializer = UniformVectorInitializer(
        objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=1, dtype=int
    )

    mutation_op = OperatorInt(
        "MutRand",
        {
            "method": "Uniform",
            "Low": objfunc.low_lim,
            "Up": objfunc.up_lim,
            "N": mutstr,
        },
    )

    search_strat = SA(
        pop_initializer,
        mutation_op,
        {"iter": n_iter, "temp_init": temp_init, "alpha": alpha},
    )

    return GeneralAlgorithm(objfunc, search_strat, params=params)


def _simulated_annealing_real_vec(objfunc, params):
    """
    Instantiates a simulated annealing algorithm to optimize the given objective function.
    This objective function should accept real coded vectors.
    """

    n_iter = params.get("iter", 100)
    temp_init = params.get("temp_init", 100)
    alpha = params.get("alpha", 0.99)
    mutstr = params.get("mutstr", 1e-5)

    pop_initializer = UniformVectorInitializer(
        objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=1, dtype=float
    )

    mutation_op = OperatorReal("RandNoise", {"method": "Gauss", "F": mutstr})

    search_strat = SA(
        pop_initializer,
        mutation_op,
        {"iter": n_iter, "temp_init": temp_init, "alpha": alpha},
    )

    return GeneralAlgorithm(objfunc, search_strat, params=params)
