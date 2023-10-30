from __future__ import annotations
from ..initializers import UniformVectorInitializer
from ..operators import OperatorInt, OperatorReal, OperatorBinary
from ..selectionMethods import SurvivorSelection, ParentSelection
from ..encodings import TypeCastEncoding
from ..strategies import ES
from ..algorithms import GeneralAlgorithm


def evolution_strategy(objfunc: ObjectiveVectorFunc, params: dict) -> Algorithm:
    """
    Instantiates a evolution strategy to optimize the given objective function.

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
        alg = _evolution_strategy_bin_vec(objfunc, params)
    elif encoding_str.lower() == "int":
        alg = _evolution_strategy_int_vec(objfunc, params)
    elif encoding_str.lower() == "real":
        alg = _evolution_strategy_real_vec(objfunc, params)
    else:
        raise ValueError(
            f'The encoding "{encoding_str}" does not exist, try "real", "int" or "bin"'
        )

    return alg


def _evolution_strategy_bin_vec(objfunc, params):
    """
    Instantiates a evolution strategy to optimize the given objective function.
    This objective function should accept binary coded vectors.
    """

    pop_size = params.get("pop_size", 100)
    offspring_size = params.get("offspring_size", 150)
    n_parents = params.get("n_parents", 100)
    mutstr = params.get("mutstr", 1)

    encoding = TypeCastEncoding(int, bool)

    pop_initializer = UniformVectorInitializer(
        objfunc.vecsize, 0, 1, pop_size=pop_size, dtype=int, encoding=encoding
    )

    cross_op = OperatorBinary("Nothing")
    mutation_op = OperatorBinary("Flip", {"N": mutstr})

    parent_sel_op = ParentSelection("Nothing")
    selection_op = SurvivorSelection("(m+n)")

    search_strat = ES(
        pop_initializer,
        mutation_op,
        cross_op,
        parent_sel_op,
        selection_op,
        {"offspringSize": offspring_size},
    )

    return GeneralAlgorithm(objfunc, search_strat, params=params)


def _evolution_strategy_int_vec(objfunc, params):
    """
    Instantiates a evolution strategy to optimize the given objective function.
    This objective function should accept integer coded vectors.
    """

    pop_size = params.get("pop_size", 100)
    offspring_size = params.get("offspring_size", 150)
    mutstr = params.get("mutstr", 1)

    pop_initializer = UniformVectorInitializer(
        objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size, dtype=int
    )

    cross_op = OperatorInt("Nothing")
    mutation_op = OperatorInt(
        "MutRand",
        {
            "method": "Uniform",
            "Low": objfunc.low_lim,
            "Up": objfunc.up_lim,
            "N": mutstr,
        },
    )

    parent_sel_op = ParentSelection("Nothing")
    selection_op = SurvivorSelection("(m+n)")

    search_strat = ES(
        pop_initializer,
        mutation_op,
        cross_op,
        parent_sel_op,
        selection_op,
        {"offspringSize": offspring_size},
    )

    return GeneralAlgorithm(objfunc, search_strat, params=params)


def _evolution_strategy_real_vec(objfunc, params):
    """
    Instantiates a evolution strategy to optimize the given objective function.
    This objective function should accept real coded vectors.
    """

    pop_size = params.get("pop_size", 100)
    offspring_size = params.get("offspring_size", 150)
    mutstr = params.get("mutstr", 1e-5)

    pop_initializer = UniformVectorInitializer(
        objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size, dtype=float
    )

    cross_op = OperatorReal("Nothing")
    mutation_op = OperatorReal("RandNoise", {"method": "Gauss", "F": mutstr})
    parent_sel_op = ParentSelection("Nothing")
    selection_op = SurvivorSelection("(m+n)")

    search_strat = ES(
        pop_initializer,
        mutation_op,
        cross_op,
        parent_sel_op,
        selection_op,
        {"offspringSize": offspring_size},
    )

    return GeneralAlgorithm(objfunc, search_strat, params=params)
