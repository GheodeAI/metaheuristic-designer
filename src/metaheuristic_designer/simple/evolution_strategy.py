from __future__ import annotations
from ..objective_function import VectorObjectiveFunc
from ..algorithm import Algorithm
from ..initializers import UniformInitializer, PermInitializer
from ..operators import create_operator
from ..operator import NullOperator
from ..survivor_selection_methods import create_survivor_selection
from ..parent_selection import NullParentSelection
from ..encodings import TypeCastEncoding
from ..strategies import ES
from ..algorithms import StandardAlgorithm


def evolution_strategy(params: dict, objfunc: VectorObjectiveFunc = None) -> Algorithm:
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
        raise ValueError('You must specify the encoding in the params structure, the options are "real", "int" and "bin"')

    encoding_str = params["encoding"]

    if encoding_str.lower() == "bin":
        alg = _evolution_strategy_bin_vec(params, objfunc)
    elif encoding_str.lower() == "int":
        alg = _evolution_strategy_int_vec(params, objfunc)
    elif encoding_str.lower() == "perm":
        alg = _evolution_strategy_perm_vec(params, objfunc)
    elif encoding_str.lower() == "real":
        alg = _evolution_strategy_real_vec(params, objfunc)
    else:
        raise ValueError(f'The encoding "{encoding_str}" does not exist, try "real", "int" or "bin"')

    return alg


def _evolution_strategy_bin_vec(params, objfunc):
    """
    Instantiates a evolution strategy to optimize the given objective function.
    This objective function should accept binary coded vectors.
    """

    pop_size = params.get("pop_size", 100)
    offspring_size = params.get("offspring_size", 150)
    mutstr = params.get("mutstr", 1)
    if objfunc is None:
        vecsize = params["vecsize"]
    else:
        vecsize = objfunc.vecsize

    encoding = TypeCastEncoding(int, bool)

    pop_initializer = UniformInitializer(vecsize, 0, 1, pop_size=pop_size, dtype=int, encoding=encoding)

    cross_op = NullOperator()
    mutation_op = create_operator("mutation.bitflip", N=mutstr)

    parent_sel_op = NullParentSelection()
    selection_op = create_survivor_selection("(m+n)")

    search_strategy = ES(pop_initializer, mutation_op, cross_op, parent_sel_op, selection_op, {"offspringSize": offspring_size})

    return StandardAlgorithm(objfunc, search_strategy, params=params)


def _evolution_strategy_int_vec(params, objfunc):
    """
    Instantiates a evolution strategy to optimize the given objective function.
    This objective function should accept integer coded vectors.
    """

    pop_size = params.get("pop_size", 100)
    offspring_size = params.get("offspring_size", 150)
    mutstr = params.get("mutstr", 1)
    if objfunc is None:
        vecsize = params["vecsize"]
    else:
        vecsize = objfunc.vecsize
    min_val = params.get("min", objfunc.low_lim if objfunc else 0)
    max_val = params.get("max", objfunc.up_lim if objfunc else 100)

    pop_initializer = UniformInitializer(vecsize, min_val, max_val, pop_size=pop_size, dtype=int)

    cross_op = NullOperator()
    mutation_op = create_operator("mutation.uniform_noise", min=objfunc.low_lim, max=objfunc.up_lim, N=mutstr)

    parent_sel_op = NullParentSelection()
    selection_op = create_survivor_selection("(m+n)")

    search_strategy = ES(pop_initializer, mutation_op, cross_op, parent_sel_op, selection_op, {"offspringSize": offspring_size})

    return StandardAlgorithm(objfunc, search_strategy, params=params)


def _evolution_strategy_perm_vec(params, objfunc):
    """
    Instantiates a evolution strategy to optimize the given objective function.
    This objective function should accept integer coded vectors.
    """

    pop_size = params.get("pop_size", 100)
    offspring_size = params.get("offspring_size", 150)
    mutstr = params.get("mutstr", 1)
    if objfunc is None:
        vecsize = params["vecsize"]
    else:
        vecsize = objfunc.vecsize

    pop_initializer = PermInitializer(vecsize, pop_size=pop_size)

    cross_op = NullOperator()
    mutation_op = create_operator("permutaiton.scramble", N=mutstr)

    parent_sel_op = NullParentSelection()
    selection_op = create_survivor_selection("(m+n)")

    search_strategy = ES(pop_initializer, mutation_op, cross_op, parent_sel_op, selection_op, {"offspringSize": offspring_size})

    return StandardAlgorithm(objfunc, search_strategy, params=params)


def _evolution_strategy_real_vec(params, objfunc):
    """
    Instantiates a evolution strategy to optimize the given objective function.
    This objective function should accept real coded vectors.
    """

    pop_size = params.get("pop_size", 100)
    offspring_size = params.get("offspring_size", 150)
    mutstr = params.get("mutstr", 1e-5)
    if objfunc is None:
        vecsize = params["vecsize"]
    else:
        vecsize = objfunc.vecsize
    min_val = params.get("min", objfunc.low_lim if objfunc else 0)
    max_val = params.get("max", objfunc.up_lim if objfunc else 100)

    pop_initializer = UniformInitializer(vecsize, min_val, max_val, pop_size=pop_size, dtype=float)

    cross_op = NullOperator()
    mutation_op = create_operator("mutation.gaussian_noise", f=mutstr)
    parent_sel_op = NullParentSelection()
    selection_op = create_survivor_selection("(m+n)")

    search_strategy = ES(pop_initializer, mutation_op, cross_op, parent_sel_op, selection_op, {"offspringSize": offspring_size})

    return StandardAlgorithm(objfunc, search_strategy, params=params)
