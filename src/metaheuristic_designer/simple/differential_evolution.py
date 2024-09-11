from __future__ import annotations
from ..initializers import UniformVectorInitializer
from ..operators import OperatorVector
from ..encodings import TypeCastEncoding
from ..strategies import DE
from ..algorithms import GeneralAlgorithm


def differential_evolution(params: dict, objfunc: ObjectiveVectorFunc = None) -> Algorithm:
    """
    Instantiates a differential evolution algorithm to optimize the given objective function.

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

    if encoding_str.lower() == "real":
        alg = _differential_evolution_real_vec(params, objfunc)
    elif encoding_str.lower() == "int":
        alg = _differential_evolution_int_vec(params, objfunc)
    elif encoding_str.lower() == "bin":
        alg = _differential_evolution_bin_vec(params, objfunc)
    else:
        raise ValueError(f'The encoding "{encoding_str}" does not exist, try "real", "int" or "bin"')

    return alg


def _differential_evolution_real_vec(params, objfunc):
    """
    Instantiates a differential evolution algorithm to optimize the given objective function.
    This objective function should accept real coded vectors.
    """

    pop_size = params.get("pop_size", 100)
    f = params.get("F", 0.8)
    cr = params.get("Cr", 0.9)
    de_type = params.get("DE_type", "de/best/1")
    if objfunc is None:
        vecsize = params["vecsize"]
    else:
        vecsize = objfunc.vecsize
    min_val = params.get("min", objfunc.low_lim if objfunc else 0)
    max_val = params.get("max", objfunc.up_lim if objfunc else 100)

    if de_type not in [
        "de/rand/1",
        "de/best/1",
        "de/rand/2",
        "de/best/2",
        "de/current-to-rand/1",
        "de/current-to-best/1",
        "de/current-to-pbest/1",
    ]:
        raise ValueError(f'Differential evolution strategy "{de_type}" does not exist.')

    pop_initializer = UniformVectorInitializer(vecsize, min_val, max_val, pop_size=pop_size, dtype=float)

    de_op = OperatorVector(de_type, {"F": f, "Cr": cr})

    search_strat = DE(pop_initializer, de_op)

    return GeneralAlgorithm(objfunc, search_strat, params=params)


def _differential_evolution_int_vec(params, objfunc):
    """
    Instantiates a differential evolution algorithm to optimize the given objective function.
    This objective function should accept real coded vectors.
    """

    pop_size = params.get("pop_size", 100)
    f = params.get("F", 0.8)
    cr = params.get("Cr", 0.9)
    de_type = params.get("DE_type", "de/best/1")
    if objfunc is None:
        vecsize = params["vecsize"]
    else:
        vecsize = objfunc.vecsize
    min_val = params.get("min", objfunc.low_lim if objfunc else 0)
    max_val = params.get("max", objfunc.up_lim if objfunc else 100)

    if de_type not in [
        "de/rand/1",
        "de/best/1",
        "de/rand/2",
        "de/best/2",
        "de/current-to-rand/1",
        "de/current-to-best/1",
        "de/current-to-pbest/1",
    ]:
        raise ValueError(f'Differential evolution strategy "{de_type}" does not exist.')

    encoding = TypeCastEncoding(float, int)

    pop_initializer = UniformVectorInitializer(
        vecsize,
        min_val,
        max_val,
        pop_size=pop_size,
        dtype=float,
        encoding=encoding,
    )

    de_op = OperatorVector(de_type, {"F": f, "Cr": cr})

    search_strat = DE(pop_initializer, de_op)

    return GeneralAlgorithm(objfunc, search_strat, params=params)


def _differential_evolution_bin_vec(params, objfunc):
    """
    Instantiates a differential evolution algorithm to optimize the given objective function.
    This objective function should accept real coded vectors.
    """

    pop_size = params.get("pop_size", 100)
    f = params.get("F", 0.8)
    cr = params.get("Cr", 0.9)
    de_type = params.get("DE_type", "de/best/1")
    if objfunc is None:
        vecsize = params["vecsize"]
    else:
        vecsize = objfunc.vecsize

    if de_type not in [
        "de/rand/1",
        "de/best/1",
        "de/rand/2",
        "de/best/2",
        "de/current-to-rand/1",
        "de/current-to-best/1",
        "de/current-to-pbest/1",
    ]:
        raise ValueError(f'Differential evolution strategy "{de_type}" does not exist.')

    encoding = TypeCastEncoding(float, bool)

    pop_initializer = UniformVectorInitializer(
        vecsize,
        0,
        1,
        pop_size=pop_size,
        dtype=float,
        encoding=encoding,
    )

    de_op = OperatorVector(de_type, {"F": f, "Cr": cr})

    search_strat = DE(pop_initializer, de_op)

    return GeneralAlgorithm(objfunc, search_strat, params=params)
