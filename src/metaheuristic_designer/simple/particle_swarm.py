from __future__ import annotations
import numpy as np
from ..objective_function import VectorObjectiveFunc
from ..algorithm import Algorithm
from ..initializer import ExtendedInitializer
from ..initializers import UniformInitializer
from ..encodings import CompositeEncoding, TypeCastEncoding, SigmoidEncoding, PSOEncoding
from ..strategies import PSO
from ..algorithms import GeneralAlgorithm
from ..constraint_handler import ExtendedConstraintHandler
from ..constraint_handlers import ClipBoundConstraint, BounceBoundConstraint


def particle_swarm(params: dict, objfunc: VectorObjectiveFunc = None) -> Algorithm:
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
    algorithm: Algorithm
        Configured optimization algorithm.
    """

    if "encoding" not in params:
        raise ValueError('You must specify the encoding in the params structure, the options are "real", "int" and "bin"')

    encoding_str = params["encoding"]

    if encoding_str.lower() == "real":
        alg = _particle_swarm_real_vec(params, objfunc)
    elif encoding_str.lower() == "int":
        alg = _particle_swarm_int_vec(params, objfunc)
    elif encoding_str.lower() == "bin":
        alg = _particle_swarm_bin_vec(params, objfunc)
    else:
        raise ValueError(f'The encoding "{encoding_str}" does not exist, try "real", "int" or "bin"')

    return alg


def _particle_swarm_real_vec(params, objfunc):
    """
    Instantiates a particle swarm algorithm to optimize the given objective function.
    This objective function should accept real coded vectors.
    """

    pop_size = params.get("pop_size", 100)
    w = params.get("w", 0.7)
    c1 = params.get("c1", 1.5)
    c2 = params.get("c2", 1.5)
    if objfunc is None:
        vecsize = params["vecsize"]
    else:
        vecsize = objfunc.vecsize

    min_val = params.get("min", objfunc.low_lim if objfunc else 0)
    max_val = params.get("max", objfunc.up_lim if objfunc else 100)
    abs_max_val = np.maximum(np.abs(min_val), np.abs(max_val))

    pso_encoding = PSOEncoding(vecsize)

    pop_initializer = ExtendedInitializer(
        solution_init=UniformInitializer(vecsize, min_val, max_val, pop_size=pop_size),
        param_init_dict={"speed": UniformInitializer(vecsize, -abs_max_val, abs_max_val)},
        encoding=pso_encoding,
    )

    constraint_handler = ExtendedConstraintHandler(
        ClipBoundConstraint(vecsize, min_val, max_val),
        {"speed": BounceBoundConstraint(vecsize, -abs_max_val, abs_max_val)},
        encoding=pso_encoding,
    )
    objfunc.constraint_handler = constraint_handler

    search_strat = PSO(initializer=pop_initializer, encoding=pso_encoding, params={"w": w, "c1": c1, "c2": c2})

    return GeneralAlgorithm(objfunc, search_strat, params=params)


def _particle_swarm_int_vec(params, objfunc):
    """
    Instantiates a particle swarm algorithm to optimize the given objective function.
    This objective function should accept real coded vectors.
    """

    pop_size = params.get("pop_size", 100)
    w = params.get("w", 0.7)
    c1 = params.get("c1", 1.5)
    c2 = params.get("c2", 1.5)
    if objfunc is None:
        vecsize = params["vecsize"]
    else:
        vecsize = objfunc.vecsize

    min_val = params.get("min", objfunc.low_lim if objfunc else 0)
    max_val = params.get("max", objfunc.up_lim if objfunc else 1)
    abs_max_val = np.maximum(np.abs(min_val), np.abs(max_val))

    pso_encoding = CompositeEncoding([
        PSOEncoding(vecsize),
        TypeCastEncoding(float, int),
    ])

    pop_initializer = ExtendedInitializer(
        solution_init=UniformInitializer(vecsize, min_val, max_val, pop_size=pop_size),
        param_init_dict={"speed": UniformInitializer(vecsize, -abs_max_val, abs_max_val)},
        encoding=pso_encoding,
    )

    constraint_handler = ExtendedConstraintHandler(
        ClipBoundConstraint(vecsize, min_val, max_val),
        {"speed": BounceBoundConstraint(vecsize, -abs_max_val, abs_max_val)},
        encoding=pso_encoding,
    )
    objfunc.constraint_handler = constraint_handler

    search_strat = PSO(initializer=pop_initializer, encoding=pso_encoding, params={"w": w, "c1": c1, "c2": c2})

    return GeneralAlgorithm(objfunc, search_strat, params=params)


def _particle_swarm_bin_vec(params, objfunc):
    """
    Instantiates a particle swarm algorithm to optimize the given objective function.
    This objective function should accept real coded vectors.
    """

    pop_size = params.get("pop_size", 100)
    w = params.get("w", 0.7)
    c1 = params.get("c1", 1.5)
    c2 = params.get("c2", 1.5)
    if objfunc is None:
        vecsize = params["vecsize"]
    else:
        vecsize = objfunc.vecsize

    min_val = params.get("min", objfunc.low_lim if objfunc else 0)
    max_val = params.get("max", objfunc.up_lim if objfunc else 1)
    abs_max_val = np.maximum(np.abs(min_val), np.abs(max_val))

    pso_encoding = CompositeEncoding([
        PSOEncoding(vecsize),
        SigmoidEncoding(as_probability=False, threshold=0.5)
    ])

    pop_initializer = ExtendedInitializer(
        solution_init=UniformInitializer(vecsize, min_val, max_val, pop_size=pop_size),
        param_init_dict={"speed": UniformInitializer(vecsize, -abs_max_val, abs_max_val)},
        encoding=pso_encoding,
    )

    constraint_handler = ExtendedConstraintHandler(
        ClipBoundConstraint(vecsize, min_val, max_val),
        {"speed": BounceBoundConstraint(vecsize, -abs_max_val, abs_max_val)},
        encoding=pso_encoding,
    )
    objfunc.constraint_handler = constraint_handler

    search_strat = PSO(initializer=pop_initializer, encoding=pso_encoding, params={"w": w, "c1": c1, "c2": c2})

    return GeneralAlgorithm(objfunc, search_strat, params=params)
