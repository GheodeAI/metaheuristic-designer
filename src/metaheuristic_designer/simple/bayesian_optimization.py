from __future__ import annotations
from ..ObjectiveFunc import VectorObjectiveFunc
from ..Algorithm import Algorithm
from ..initializers import UniformVectorInitializer
from ..selectionMethods import ParentSelectionNull
from ..strategies import BayesianOptimization
from ..algorithms import GeneralAlgorithm


def bayesian_optimization(params: dict, objfunc: VectorObjectiveFunc = None) -> Algorithm:
    """
    Instantiates a bayesian optimization algorithm to optimize the given objective function.

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
        raise ValueError('You must specify the encoding in the params structure, the algorithm is just implemented for the "real" encoding.')

    encoding_str = params["encoding"]

    if encoding_str.lower() == "real":
        alg = _bayesian_optimization_real_vec(params, objfunc)
    else:
        raise ValueError(f'The encoding "{encoding_str}" does not exist, try "real"')

    return alg


def _bayesian_optimization_real_vec(params, objfunc):
    """
    Instantiates a bayesian optimization algorithm to optimize the given objective function.
    This objective function should accept binary coded vectors.
    """

    pop_size = params.get("pop_size", 100)
    if objfunc is None:
        vecsize = params["vecsize"]
    else:
        vecsize = objfunc.vecsize
    min_val = params.get("min", objfunc.low_lim if objfunc else -100)
    max_val = params.get("max", objfunc.up_lim if objfunc else 100)

    search_strat = BayesianOptimization(
        initializer=UniformVectorInitializer(vecsize, min_val, max_val, pop_size=pop_size),
        params=params,
    )

    return GeneralAlgorithm(objfunc, search_strat, params=params)
