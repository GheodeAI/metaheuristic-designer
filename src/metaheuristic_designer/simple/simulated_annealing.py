from __future__ import annotations
import numpy as np
from ..initializers import UniformInitializer, PermInitializer
from ..encodings import TypeCastEncoding
from ..strategies import SA
from ..algorithms import StandardAlgorithm
from ..operators import create_operator

def simulated_annealing_binary(objfunc, mutated_bits=1, initial_temperature=1.0, alpha=0.997, iterations=100, encoding=None, **kwargs):
    """
    Instantiates a hill climbing algorithm to optimize the given objective function.
    This objective function should accept binary coded vectors.
    """

    encoding = TypeCastEncoding(int, bool) if encoding is None else encoding
    pop_initializer = UniformInitializer(objfunc.vecsize, 0, 1, pop_size=1, dtype=np.uint8, encoding=encoding)
    mutation_op = create_operator("mutation.bitflip", N=mutated_bits)
    search_strat = SA(pop_initializer, mutation_op, temperature_init=initial_temperature, alpha=alpha, iterations=iterations)
    return StandardAlgorithm(objfunc, search_strat, **kwargs)


def simulated_annealing_permutation(objfunc, swapped_positions=2, initial_temperature=1.0, alpha=0.997, iterations=100, encoding=None, **kwargs):
    """
    Instantiates a hill climbing algorithm to optimize the given objective function.
    This objective function should accept integer coded vectors.
    """

    pop_initializer = PermInitializer(objfunc.vecsize, pop_size=1, encoding=encoding)
    mutation_op = create_operator("permutation.swap", N=swapped_positions)
    search_strat = SA(pop_initializer, mutation_op, temperature_init=initial_temperature, alpha=alpha, iterations=iterations)
    return StandardAlgorithm(objfunc, search_strat, **kwargs)


def simulated_annealing_discrete(objfunc, resampled_components=1, initial_temperature=1.0, alpha=0.997, iterations=100, encoding=None, **kwargs):
    """
    Instantiates a hill climbing algorithm to optimize the given objective function.
    This objective function should accept integer coded vectors.
    """

    pop_initializer = UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=1, dtype=int, encoding=encoding)
    mutation_op = create_operator("random.reset", N=resampled_components)
    search_strat = SA(pop_initializer, mutation_op, temperature_init=initial_temperature, alpha=alpha, iterations=iterations)
    return StandardAlgorithm(objfunc, search_strat, **kwargs)


def simulated_annealing_real(objfunc, mutation_strength=1e-5, mutated_components=1, initial_temperature=1.0, alpha=0.997, iterations=100, encoding=None, **kwargs):
    """
    Instantiates a hill climbing algorithm to optimize the given objective function.
    This objective function should accept real coded vectors.
    """

    pop_initializer = UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=1, dtype=float, encoding=encoding)
    mutation_op = create_operator("mutation.gaussian_mutation", F=mutation_strength, N=mutated_components)
    search_strat = SA(pop_initializer, mutation_op, temperature_init=initial_temperature, alpha=alpha, iterations=iterations)
    return StandardAlgorithm(objfunc, search_strat, **kwargs)

