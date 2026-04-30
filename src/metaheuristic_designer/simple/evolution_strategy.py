from __future__ import annotations
import numpy as np

from metaheuristic_designer import survivor_selection
from ..initializers import UniformInitializer, PermInitializer
from ..encodings import TypeCastEncoding
from ..strategies import ES
from ..algorithms import StandardAlgorithm
from ..operators import create_operator
from ..survivor_selection_methods import create_survivor_selection
from ..utils import check_random_state

def hill_climb_binary(objfunc, mutated_bits=1, population_size=100, offspring_size=500, encoding=None, random_state=None, **kwargs):
    """
    Instantiates a hill climbing algorithm to optimize the given objective function.
    This objective function should accept binary coded vectors.
    """

    random_state = check_random_state(random_state)
    encoding = TypeCastEncoding(int, bool) if encoding is None else encoding
    pop_initializer = UniformInitializer(objfunc.vecsize, 0, 1, pop_size=population_size, dtype=np.uint8, encoding=encoding, random_state=random_state)
    mutation_op = create_operator("mutation.bitflip", N=mutated_bits, random_state=random_state)
    survivor_sel = create_survivor_selection("keep_best", random_state=random_state)
    search_strat = ES(pop_initializer, mutation_op, survivor_selection=survivor_sel, offspring_size=offspring_size, random_state=random_state)
    return StandardAlgorithm(objfunc, search_strat, **kwargs)


def hill_climb_permutation(objfunc, swapped_positions=2, population_size=100, offspring_size=500, encoding=None, random_state=None, **kwargs):
    """
    Instantiates a hill climbing algorithm to optimize the given objective function.
    This objective function should accept integer coded vectors.
    """

    random_state = check_random_state(random_state)
    pop_initializer = PermInitializer(objfunc.vecsize, pop_size=population_size, encoding=encoding, random_state=random_state)
    mutation_op = create_operator("permutation.swap", N=swapped_positions, random_state=random_state)
    survivor_sel = create_survivor_selection("keep_best", random_state=random_state)
    search_strat = ES(pop_initializer, mutation_op, survivor_selection=survivor_sel, offspring_size=offspring_size, random_state=random_state)
    return StandardAlgorithm(objfunc, search_strat, **kwargs)


def hill_climb_discrete(objfunc, resampled_components=1, population_size=100, offspring_size=500, encoding=None, random_state=None, **kwargs):
    """
    Instantiates a hill climbing algorithm to optimize the given objective function.
    This objective function should accept integer coded vectors.
    """

    random_state = check_random_state(random_state)
    pop_initializer = UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=population_size, dtype=int, encoding=encoding, random_state=random_state)
    mutation_op = create_operator("random.reset", N=resampled_components, random_state=random_state)
    survivor_sel = create_survivor_selection("keep_best", random_state=random_state)
    search_strat = ES(pop_initializer, mutation_op, survivor_selection=survivor_sel, offspring_size=offspring_size, random_state=random_state)
    return StandardAlgorithm(objfunc, search_strat, **kwargs)


def hill_climb_real(objfunc, mutation_strength=1e-5, mutated_components=1, population_size=100, offspring_size=500, encoding=None, random_state=None, **kwargs):
    """
    Instantiates a hill climbing algorithm to optimize the given objective function.
    This objective function should accept real coded vectors.
    """

    random_state = check_random_state(random_state)
    pop_initializer = UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=population_size, dtype=float, encoding=encoding, random_state=random_state)
    mutation_op = create_operator("mutation.gaussian_mutation", F=mutation_strength, N=mutated_components, random_state=random_state)
    survivor_sel = create_survivor_selection("keep_best", random_state=random_state)
    search_strat = ES(pop_initializer, mutation_op, survivor_selection=survivor_sel, offspring_size=offspring_size, random_state=random_state)
    return StandardAlgorithm(objfunc, search_strat, **kwargs)
