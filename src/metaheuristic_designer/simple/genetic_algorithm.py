from __future__ import annotations
from ..objective_function import VectorObjectiveFunc
from ..algorithm import Algorithm
from ..initializers import UniformInitializer, PermInitializer
from ..operators import create_operator, create_permutation_operator, create_crossover_operator
from ..parent_selection import create_parent_selection
from ..survivor_selection import create_survivor_selection
from ..encodings import TypeCastEncoding
from ..strategies import GA


def genetic_algorithm(params: dict, objfunc: VectorObjectiveFunc = None) -> Algorithm:
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
        raise ValueError('You must specify the encoding in the params structure, the options are "real", "int" and "bin"')

    encoding_str = params["encoding"]

    if encoding_str.lower() == "bin":
        alg = _genetic_algorithm_bin_vec(params, objfunc)
    elif encoding_str.lower() == "int":
        alg = _genetic_algorithm_int_vec(params, objfunc)
    elif encoding_str.lower() == "perm":
        alg = _genetic_algorithm_perm_vec(params, objfunc)
    elif encoding_str.lower() == "real":
        alg = _genetic_algorithm_real_vec(params, objfunc)
    else:
        raise ValueError(f'The encoding "{encoding_str}" does not exist, try "real", "int", "perm" or "bin"')

    return alg


def _genetic_algorithm_bin_vec(params, objfunc):
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
    if objfunc is None:
        vecsize = params["vecsize"]
    else:
        vecsize = objfunc.vecsize

    encoding = TypeCastEncoding(int, bool)

    pop_initializer = UniformInitializer(vecsize, 0, 1, pop_size=pop_size, dtype=int, encoding=encoding)

    cross_op = create_crossover_operator(cross_method)
    mutation_op = create_operator("mutation.bitflip", N=mutstr)

    parent_sel_op = create_parent_selection("best", amount=n_parents)
    selection_op = create_survivor_selection("keep_best")

    search_strategy = GA(pop_initializer, mutation_op, cross_op, parent_sel_op, selection_op, {"pcross": pcross, "pmut": pmut})

    return Algorithm(objfunc, search_strategy, params=params)


def _genetic_algorithm_int_vec(params, objfunc):
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
    if objfunc is None:
        vecsize = params["vecsize"]
    else:
        vecsize = objfunc.vecsize
    min_val = params.get("min", objfunc.low_lim if objfunc else 0)
    max_val = params.get("max", objfunc.up_lim if objfunc else 100)

    pop_initializer = UniformInitializer(vecsize, min_val, max_val, pop_size=pop_size, dtype=int)

    cross_op = create_crossover_operator(cross_method)
    mutation_op = create_operator(
        "mutation.uniform_mutation",
        min=objfunc.low_lim if objfunc is not None else min_val,
        max=objfunc.up_lim if objfunc is not None else max_val,
        N=mutstr,
    )

    parent_sel_op = create_parent_selection("best", amount=n_parents)
    selection_op = create_survivor_selection("keep_best")

    search_strategy = GA(pop_initializer, mutation_op, cross_op, parent_sel_op, selection_op, {"pcross": pcross, "pmut": pmut})

    return Algorithm(objfunc, search_strategy, params=params)


def _genetic_algorithm_perm_vec(params, objfunc):
    """
    Instantiates a genetic algorithm to optimize the given objective function.
    This objective function should accept integer coded vectors.
    """

    pop_size = params.get("pop_size", 100)
    n_parents = params.get("n_parents", 20)
    cross_method = params.get("cross", "OrderCross")
    pcross = params.get("pcross", 0.8)
    pmut = params.get("pmut", 0.1)
    mutstr = params.get("mutstr", 1)
    if objfunc is None:
        vecsize = params["vecsize"]
    else:
        vecsize = objfunc.vecsize

    pop_initializer = PermInitializer(vecsize, pop_size=pop_size)

    cross_op = create_permutation_operator(cross_method)
    mutation_op = create_permutation_operator("scramble", N=mutstr)

    parent_sel_op = create_parent_selection("best", amount=n_parents)
    selection_op = create_survivor_selection("keep_best")

    search_strategy = GA(pop_initializer, mutation_op, cross_op, parent_sel_op, selection_op, {"pcross": pcross, "pmut": pmut})

    return Algorithm(objfunc, search_strategy, params=params)


def _genetic_algorithm_real_vec(params, objfunc):
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
    if objfunc is None:
        vecsize = params["vecsize"]
    else:
        vecsize = objfunc.vecsize
    min_val = params.get("min", objfunc.low_lim if objfunc else 0)
    max_val = params.get("max", objfunc.up_lim if objfunc else 100)

    pop_initializer = UniformInitializer(vecsize, min_val, max_val, pop_size=pop_size, dtype=float)

    cross_op = create_operator(cross_method)
    mutation_op = create_operator("mutation.gaussian_mutation", f=mutstr, N=1)

    parent_sel_op = create_parent_selection("best", amount=n_parents)
    selection_op = create_survivor_selection("keep_best")

    search_strategy = GA(pop_initializer, mutation_op, cross_op, parent_sel_op, selection_op, {"pcross": pcross, "pmut": pmut})

    return Algorithm(objfunc, search_strategy, params=params)
