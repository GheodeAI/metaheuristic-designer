from __future__ import annotations
import numpy as np
from ..initializers import UniformInitializer, PermInitializer
from ..encodings import TypeCastEncoding
from ..strategies import LocalSearch
from ..algorithms import Algorithm
from ..operators import create_operator
from ..utils import check_random_state


def local_search_binary(objfunc, mutated_bits=1, samples_per_iteration=100, encoding=None, random_state=None, **kwargs):
    """
    Instantiates a hill climbing algorithm to optimize the given objective function.
    This objective function should accept binary coded vectors.
    """

    random_state = check_random_state(random_state)
    encoding = TypeCastEncoding(int, bool) if encoding is None else encoding
    pop_initializer = UniformInitializer(objfunc.vecsize, 0, 1, pop_size=1, dtype=np.uint8, encoding=encoding, random_state=random_state)
    mutation_op = create_operator("mutation.bitflip", N=mutated_bits, random_state=random_state)
    search_strat = LocalSearch(pop_initializer, mutation_op, iterations=samples_per_iteration, random_state=random_state)
    return Algorithm(objfunc, search_strat, **kwargs)


def local_search_permutation(objfunc, swapped_positions=2, samples_per_iteration=100, encoding=None, random_state=None, **kwargs):
    """
    Instantiates a hill climbing algorithm to optimize the given objective function.
    This objective function should accept integer coded vectors.
    """

    random_state = check_random_state(random_state)
    pop_initializer = PermInitializer(objfunc.vecsize, pop_size=1, encoding=encoding, random_state=random_state)
    mutation_op = create_operator("permutation.swap", N=swapped_positions, random_state=random_state)
    search_strat = LocalSearch(pop_initializer, mutation_op, iterations=samples_per_iteration, random_state=random_state)
    return Algorithm(objfunc, search_strat, **kwargs)


def local_search_discrete(objfunc, resampled_components=1, samples_per_iteration=100, encoding=None, random_state=None, **kwargs):
    """
    Instantiates a hill climbing algorithm to optimize the given objective function.
    This objective function should accept integer coded vectors.
    """

    random_state = check_random_state(random_state)
    pop_initializer = UniformInitializer(
        objfunc.vecsize, objfunc.lower_bound, objfunc.upper_bound, pop_size=1, dtype=int, encoding=encoding, random_state=random_state
    )
    mutation_op = create_operator("random.reset", n=resampled_components, random_state=random_state)
    search_strat = LocalSearch(pop_initializer, mutation_op, iterations=samples_per_iteration, random_state=random_state)
    return Algorithm(objfunc, search_strat, **kwargs)


def local_search_real(objfunc, mutation_strength=1e-2, mutated_components=1, samples_per_iteration=100, encoding=None, random_state=None, **kwargs):
    """
    Instantiates a hill climbing algorithm to optimize the given objective function.
    This objective function should accept real coded vectors.
    """

    random_state = check_random_state(random_state)
    pop_initializer = UniformInitializer(
        objfunc.vecsize, objfunc.lower_bound, objfunc.upper_bound, pop_size=1, dtype=float, encoding=encoding, random_state=random_state
    )
    mutation_op = create_operator("mutation.gaussian_mutation", F=mutation_strength, N=mutated_components, random_state=random_state)
    search_strat = LocalSearch(pop_initializer, mutation_op, iterations=samples_per_iteration, random_state=random_state)
    return Algorithm(objfunc, search_strat, **kwargs)
