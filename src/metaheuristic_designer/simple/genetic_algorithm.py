from __future__ import annotations
from ..initializers import UniformVectorInitializer
from ..operators import OperatorInt, OperatorReal, OperatorBinary
from ..selectionMethods import SurvivorSelection, ParentSelection
from ..encodings import TypeCastEncoding
from ..strategies import GA
from ..algorithms import GeneralAlgorithm


def genetic_algorithm(objfunc: ObjectiveVectorFunc, params: dict) -> Algorithm:
    """
    Instantiates a genetic algorithm to optimize the given objective function.

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
        alg = _genetic_algorithm_bin_vec(objfunc, params)
    elif encoding_str.lower() == "int":
        alg = _genetic_algorithm_int_vec(objfunc, params)
    elif encoding_str.lower() == "real":
        alg = _genetic_algorithm_real_vec(objfunc, params)
    else:
        raise ValueError(
            f'The encoding "{encoding_str}" does not exist, try "real", "int" or "bin"'
        )

    return alg


def _genetic_algorithm_bin_vec(objfunc, params):
    """
    Instantiates a genetic algorithm to optimize the given objective function.
    This objective function should accept binary coded vectors.
    """

    pop_size = params.get("pop_size", 100)
    n_parents = params.get("n_parents", 20)
    cross_method = params.get("cross", "Multipoint")
    pcross = params.get("pcross", 0.8)
    pmut = params.get("pmut", 0.1)
    mutstr = params.get("mutstr", 1)

    encoding = TypeCastEncoding(int, bool)

    pop_initializer = UniformVectorInitializer(
        objfunc.vecsize, 0, 1, pop_size=pop_size, dtype=int, encoding=encoding
    )

    cross_op = OperatorBinary(cross_method)
    mutation_op = OperatorBinary("Flip", {"N": mutstr})

    parent_sel_op = ParentSelection("Best", {"amount": n_parents})
    selection_op = SurvivorSelection("KeepBest")

    search_strat = GA(
        pop_initializer,
        mutation_op,
        cross_op,
        parent_sel_op,
        selection_op,
        {"pcross": pcross, "pmut": pmut},
    )

    return GeneralAlgorithm(objfunc, search_strat, params=params)


def _genetic_algorithm_int_vec(objfunc, params):
    """
    Instantiates a genetic algorithm to optimize the given objective function.
    This objective function should accept integer coded vectors.
    """

    pop_size = params.get("pop_size", 100)
    n_parents = params.get("n_parents", 20)
    cross_method = params.get("cross", "Multipoint")
    pcross = params.get("pcross", 0.8)
    pmut = params.get("pmut", 0.1)
    mutstr = params.get("mutstr", 1)

    pop_initializer = UniformVectorInitializer(
        objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size, dtype=int
    )

    cross_op = OperatorInt(cross_method)
    mutation_op = OperatorInt(
        "MutRand",
        {
            "method": "Uniform",
            "Low": objfunc.low_lim,
            "Up": objfunc.up_lim,
            "N": mutstr,
        },
    )

    parent_sel_op = ParentSelection("Best", {"amount": n_parents})
    selection_op = SurvivorSelection("KeepBest")

    search_strat = GA(
        pop_initializer,
        mutation_op,
        cross_op,
        parent_sel_op,
        selection_op,
        {"pcross": pcross, "pmut": pmut},
    )

    return GeneralAlgorithm(objfunc, search_strat, params=params)


def _genetic_algorithm_real_vec(objfunc, params):
    """
    Instantiates a genetic algorithm to optimize the given objective function.
    This objective function should accept real coded vectors.
    """

    pop_size = params.get("pop_size", 100)
    n_parents = params.get("n_parents", 20)
    cross_method = params.get("cross", "Multipoint")
    pcross = params.get("pcross", 0.8)
    pmut = params.get("pmut", 0.1)
    mutstr = params.get("mutstr", 1e-5)

    pop_initializer = UniformVectorInitializer(
        objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size, dtype=float
    )

    cross_op = OperatorReal(cross_method)
    mutation_op = OperatorReal("RandNoise", {"method": "Gauss", "F": mutstr, "N": 1})

    parent_sel_op = ParentSelection("Best", {"amount": n_parents})
    selection_op = SurvivorSelection("KeepBest")

    search_strat = GA(
        pop_initializer,
        mutation_op,
        cross_op,
        parent_sel_op,
        selection_op,
        {"pcross": pcross, "pmut": pmut},
    )

    return GeneralAlgorithm(objfunc, search_strat, params=params)
