"""
Ready-to-run Hill Climbing wrappers.
"""

from __future__ import annotations
from typing import Optional
import numpy as np

from metaheuristic_designer.encoding import Encoding
from metaheuristic_designer.objective_function import ObjectiveFunc
from ..initializers import UniformInitializer, PermInitializer
from ..encodings import TypeCastEncoding
from ..strategies import HillClimb
from ..algorithms import Algorithm
from ..operators import create_operator
from ..utils import RNGLike, check_random_state


def hill_climb_binary(
    objfunc: ObjectiveFunc, mutated_bits: int = 1, encoding: Optional[Encoding] = None, random_state: Optional[RNGLike] = None, **kwargs
) -> Algorithm:
    """Hill Climbing for binary-coded vectors.

    Parameters
    ----------
    objfunc : ObjectiveFunc
        The objective function to optimize.
    mutated_bits : int, optional
        Number of bits flipped per mutation (default 1).
    encoding : Encoding, optional
        Encoding; defaults to :class:`TypeCastEncoding` (int → bool).
    random_state : RNGLike, optional
        Random seed or generator.
    **kwargs
        Forwarded to :class:`Algorithm`.
    """

    random_state = check_random_state(random_state)
    encoding = TypeCastEncoding(int, bool) if encoding is None else encoding
    pop_initializer = UniformInitializer(objfunc.dimension, 0, 1, population_size=1, dtype=np.uint8, encoding=encoding, random_state=random_state)
    mutation_op = create_operator("mutation.bitflip", N=mutated_bits, random_state=random_state)
    search_strat = HillClimb(pop_initializer, mutation_op, random_state=random_state)
    return Algorithm(objfunc, search_strat, **kwargs)


def hill_climb_permutation(
    objfunc: ObjectiveFunc, swapped_positions: int = 2, encoding: Optional[Encoding] = None, random_state: Optional[RNGLike] = None, **kwargs
) -> Algorithm:
    """Hill Climbing for permutation-coded vectors.

    Parameters
    ----------
    objfunc : ObjectiveFunc
        The objective function to optimize.
    swapped_positions : int, optional
        Number of positions swapped per mutation (default 2).
    encoding : Encoding, optional
        Encoding applied to the genotype.
    random_state : RNGLike, optional
        Random seed or generator.
    **kwargs
        Forwarded to :class:`Algorithm`.
    """

    random_state = check_random_state(random_state)
    pop_initializer = PermInitializer(objfunc.dimension, population_size=1, encoding=encoding, random_state=random_state)
    mutation_op = create_operator("permutation.swap", N=swapped_positions, random_state=random_state)
    search_strat = HillClimb(pop_initializer, mutation_op, random_state=random_state)
    return Algorithm(objfunc, search_strat, **kwargs)


def hill_climb_discrete(
    objfunc: ObjectiveFunc, resampled_components: int = 1, encoding: Optional[Encoding] = None, random_state: Optional[RNGLike] = None, **kwargs
) -> Algorithm:
    """Hill Climbing for integer-coded vectors.

    Parameters
    ----------
    objfunc : ObjectiveFunc
        The objective function to optimize.
    resampled_components : int, optional
        Number of components resampled per mutation (default 1).
    encoding : Encoding, optional
        Encoding applied to the genotype.
    random_state : RNGLike, optional
        Random seed or generator.
    **kwargs
        Forwarded to :class:`Algorithm`.
    """

    random_state = check_random_state(random_state)
    pop_initializer = UniformInitializer(
        objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=1, dtype=int, encoding=encoding, random_state=random_state
    )
    mutation_op = create_operator("random.reset", n=resampled_components, random_state=random_state)
    search_strat = HillClimb(pop_initializer, mutation_op, random_state=random_state)
    return Algorithm(objfunc, search_strat, **kwargs)


def hill_climb_real(
    objfunc: ObjectiveFunc,
    mutation_strength: float = 1e-2,
    mutated_components: int = 1,
    encoding: Optional[Encoding] = None,
    random_state: Optional[RNGLike] = None,
    **kwargs,
) -> Algorithm:
    """Hill Climbing for real-coded vectors.

    Parameters
    ----------
    objfunc : ObjectiveFunc
        The objective function to optimize.
    mutation_strength : float, optional
        Standard deviation of Gaussian mutation (default 1e-2).
    mutated_components : int, optional
        Number of components mutated per individual (default 1).
    encoding : Encoding, optional
        Encoding applied to the genotype.
    random_state : RNGLike, optional
        Random seed or generator.
    **kwargs
        Forwarded to :class:`Algorithm`.
    """

    random_state = check_random_state(random_state)
    pop_initializer = UniformInitializer(
        objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=1, dtype=float, encoding=encoding, random_state=random_state
    )
    mutation_op = create_operator("mutation.gaussian_mutation", F=mutation_strength, N=mutated_components, random_state=random_state)
    search_strat = HillClimb(pop_initializer, mutation_op, random_state=random_state)
    return Algorithm(objfunc, search_strat, **kwargs)
