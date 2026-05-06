from __future__ import annotations
import numpy as np

from ..algorithm import Algorithm
from ..initializers import UniformInitializer, PermInitializer
from ..operators import create_operator
from ..parent_selection import create_parent_selection
from ..survivor_selection import create_survivor_selection
from ..encodings import TypeCastEncoding
from ..strategies import GA
from ..utils import check_random_state


def genetic_algorithm_binary(objfunc, mutated_bits=1, population_size=100, encoding=None, random_state=None, **kwargs):
    """
    Instantiates a hill climbing algorithm to optimize the given objective function.
    This objective function should accept binary coded vectors.
    """

    random_state = check_random_state(random_state)
    encoding = TypeCastEncoding(int, bool) if encoding is None else encoding
    pop_initializer = UniformInitializer(
        objfunc.dimension, 0, 1, pop_size=population_size, dtype=np.uint8, encoding=encoding, random_state=random_state
    )
    mutation_op = create_operator("mutation.bitflip", N=mutated_bits, random_state=random_state)
    crossover_op = create_operator("crossover.multipoint", random_state=random_state)
    parent_sel = create_parent_selection("tournament", amount=20, random_state=random_state)
    survivor_sel = create_survivor_selection("elitism", amount=10, random_state=random_state)
    search_strat = GA(
        pop_initializer,
        mutation_op=mutation_op,
        crossover_op=crossover_op,
        parent_sel=parent_sel,
        survivor_sel=survivor_sel,
        mutation_prob=0.1,
        crossover_prob=0.8,
        random_state=random_state,
    )
    return Algorithm(objfunc, search_strat, **kwargs)


def genetic_algorithm_permutation(objfunc, swapped_positions=2, population_size=100, encoding=None, random_state=None, **kwargs):
    """
    Instantiates a hill climbing algorithm to optimize the given objective function.
    This objective function should accept integer coded vectors.
    """

    random_state = check_random_state(random_state)
    pop_initializer = PermInitializer(objfunc.dimension, population_size=population_size, encoding=encoding, random_state=random_state)
    mutation_op = create_operator("permutation.swap", N=swapped_positions, random_state=random_state)
    crossover_op = create_operator("crossover.multipoint", random_state=random_state)
    parent_sel = create_parent_selection("tournament", amount=20, random_state=random_state)
    survivor_sel = create_survivor_selection("elitism", amount=10, random_state=random_state)
    search_strat = GA(
        pop_initializer,
        mutation_op=mutation_op,
        crossover_op=crossover_op,
        parent_sel=parent_sel,
        survivor_sel=survivor_sel,
        mutation_prob=0.1,
        crossover_prob=0.8,
        random_state=random_state,
    )
    return Algorithm(objfunc, search_strat, **kwargs)


def genetic_algorithm_discrete(objfunc, resampled_components=1, population_size=100, encoding=None, random_state=None, **kwargs):
    """
    Instantiates a hill climbing algorithm to optimize the given objective function.
    This objective function should accept integer coded vectors.
    """

    random_state = check_random_state(random_state)
    pop_initializer = UniformInitializer(
        objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, pop_size=population_size, dtype=int, encoding=encoding, random_state=random_state
    )
    mutation_op = create_operator("random.reset", n=resampled_components, random_state=random_state)
    crossover_op = create_operator("crossover.multipoint", random_state=random_state)
    parent_sel = create_parent_selection("tournament", amount=20, random_state=random_state)
    survivor_sel = create_survivor_selection("elitism", amount=10, random_state=random_state)
    search_strat = GA(
        pop_initializer,
        mutation_op=mutation_op,
        crossover_op=crossover_op,
        parent_sel=parent_sel,
        survivor_sel=survivor_sel,
        mutation_prob=0.1,
        crossover_prob=0.8,
        random_state=random_state,
    )
    return Algorithm(objfunc, search_strat, **kwargs)


def genetic_algorithm_real(objfunc, mutation_strength=1e-2, mutated_components=1, population_size=100, encoding=None, random_state=None, **kwargs):
    """
    Instantiates a hill climbing algorithm to optimize the given objective function.
    This objective function should accept real coded vectors.
    """

    random_state = check_random_state(random_state)
    pop_initializer = UniformInitializer(
        objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, pop_size=population_size, dtype=float, encoding=encoding, random_state=random_state
    )
    mutation_op = create_operator("mutation.gaussian_mutation", F=mutation_strength, N=mutated_components, random_state=random_state)
    crossover_op = create_operator("crossover.multipoint", random_state=random_state)
    parent_sel = create_parent_selection("tournament", amount=20, random_state=random_state)
    survivor_sel = create_survivor_selection("elitism", amount=10, random_state=random_state)
    search_strat = GA(
        pop_initializer,
        mutation_op=mutation_op,
        crossover_op=crossover_op,
        parent_sel=parent_sel,
        survivor_sel=survivor_sel,
        mutation_prob=0.1,
        crossover_prob=0.8,
        random_state=random_state,
    )
    return Algorithm(objfunc, search_strat, **kwargs)
