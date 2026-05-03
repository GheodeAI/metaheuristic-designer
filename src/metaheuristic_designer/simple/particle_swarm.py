from __future__ import annotations
import numpy as np
from ..objective_function import VectorObjectiveFunc
from ..algorithm import Algorithm
from ..initializers import UniformInitializer, ExtendedInitializer
from ..encodings import CompositeEncoding, TypeCastEncoding, SigmoidEncoding, PSOEncoding
from ..strategies import PSO
from ..algorithms import Algorithm
from ..constraint_handlers import ClipBoundConstraint, BounceBoundConstraint, ExtendedConstraintHandler

particle_swarm_binary = lambda *args, **kwargs: None
particle_swarm_discrete = lambda *args, **kwargs: None
particle_swarm_real = lambda *args, **kwargs: None

def hill_climb_binary(objfunc, mutated_bits=1, encoding=None, random_state=None, **kwargs):
    """
    Instantiates a hill climbing algorithm to optimize the given objective function.
    This objective function should accept binary coded vectors.
    """

    random_state = check_random_state(random_state)
    # encoding = TypeCastEncoding(int, bool) if encoding is None else encoding
    # pop_initializer = UniformInitializer(objfunc.vecsize, 0, 1, pop_size=1, dtype=np.uint8, encoding=encoding, random_state=random_state)
    # mutation_op = create_operator("mutation.bitflip", N=mutated_bits, random_state=random_state)
    # search_strat = HillClimb(pop_initializer, mutation_op, random_state=random_state)
    # return Algorithm(objfunc, search_strat, **kwargs)
    encoding = PSOEncoding(objfunc.vecsize)
    base_constraint_handler = objfunc.constraint_handler
    objfunc.constraint_handler = ExtendedConstraintHandler(
        solution_handler=base_constraint_handler,
        param_handler_dict={"speed": BounceBoundConstraint(objfunc.vecsize)},
        encoding=encoding
    )
    abs_up_lim = np.maximum(np.abs(objfunc.low_lim), np.abs(objfunc.up_lim))
    initializer = ExtendedInitializer(
        solution_init=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size, random_state=random_state),
        param_init_dict={"speed": UniformInitializer(objfunc.vecsize, -abs_up_lim, abs_up_lim)},
        encoding=encoding,
    )
    search_strategy = PSO(
        initializer=initializer,
        encoding=encoding,
        w=0.7,
        c1=1.5,
        c2=1.5
    )


def hill_climb_discrete(objfunc, resampled_components=1, encoding=None, random_state=None, **kwargs):
    """
    Instantiates a hill climbing algorithm to optimize the given objective function.
    This objective function should accept integer coded vectors.
    """

    random_state = check_random_state(random_state)
    pop_initializer = UniformInitializer(
        objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=1, dtype=int, encoding=encoding, random_state=random_state
    )
    mutation_op = create_operator("random.reset", n=resampled_components, random_state=random_state)
    search_strat = HillClimb(pop_initializer, mutation_op, random_state=random_state)
    return Algorithm(objfunc, search_strat, **kwargs)


def hill_climb_real(objfunc, mutation_strength=1e-2, mutated_components=1, encoding=None, random_state=None, **kwargs):
    """
    Instantiates a hill climbing algorithm to optimize the given objective function.
    This objective function should accept real coded vectors.
    """

    random_state = check_random_state(random_state)
    pop_initializer = UniformInitializer(
        objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=1, dtype=float, encoding=encoding, random_state=random_state
    )
    mutation_op = create_operator("mutation.gaussian_mutation", F=mutation_strength, N=mutated_components, random_state=random_state)
    search_strat = HillClimb(pop_initializer, mutation_op, random_state=random_state)
    return Algorithm(objfunc, search_strat, **kwargs)
