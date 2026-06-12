"""
Ready-to-run Local Search wrappers.
"""

from __future__ import annotations
from typing import Optional
import numpy as np

from metaheuristic_designer.encoding import Encoding
from metaheuristic_designer.objective_function import ObjectiveFunc
from ..initializers import UniformInitializer, PermInitializer
from ..encodings import TypeCastEncoding
from ..strategies import LocalSearch
from ..algorithms import Algorithm
from ..operators import create_operator
from ..utils import RNGLike, check_rng


def local_search_binary(
    objfunc: ObjectiveFunc,
    mutated_bits: int = 1,
    samples_per_iteration: int = 100,
    encoding: Optional[Encoding] = None,
    rng: Optional[RNGLike] = None,
    **kwargs,
) -> Algorithm:
    """Local Search for binary-coded vectors.

    Parameters
    ----------
    objfunc : ObjectiveFunc
        The objective function to optimize.
    mutated_bits : int, optional
        Number of bits flipped per mutation (default 1).
    samples_per_iteration : int, optional
        Number of samples evaluated per iteration (default 100).
    encoding : Encoding, optional
        Encoding; defaults to :class:`TypeCastEncoding` (int → bool).
    rng : RNGLike, optional
        Random seed or generator.
    **kwargs
        Forwarded to :class:`Algorithm`.
    """

    rng = check_rng(rng)
    encoding = TypeCastEncoding(int, bool) if encoding is None else encoding
    pop_initializer = UniformInitializer(objfunc.dimension, 0, 1, population_size=1, dtype=np.uint8, encoding=encoding, rng=rng)
    mutation_op = create_operator("mutation.bitflip", N=mutated_bits, rng=rng)
    search_strat = LocalSearch(pop_initializer, mutation_op, iterations=samples_per_iteration, rng=rng)
    return Algorithm(objfunc, search_strat, **kwargs)


def local_search_permutation(
    objfunc: ObjectiveFunc,
    swapped_positions: int = 2,
    samples_per_iteration: int = 100,
    encoding: Optional[Encoding] = None,
    rng: Optional[RNGLike] = None,
    **kwargs,
) -> Algorithm:
    """Local Search for permutation-coded vectors.

    Parameters
    ----------
    objfunc : ObjectiveFunc
        The objective function to optimize.
    swapped_positions : int, optional
        Number of positions swapped per mutation (default 2).
    samples_per_iteration : int, optional
        Number of samples evaluated per iteration (default 100).
    encoding : Encoding, optional
        Encoding applied to the genotype.
    rng : RNGLike, optional
        Random seed or generator.
    **kwargs
        Forwarded to :class:`Algorithm`.
    """

    rng = check_rng(rng)
    pop_initializer = PermInitializer(objfunc.dimension, population_size=1, encoding=encoding, rng=rng)
    mutation_op = create_operator("permutation.swap", N=swapped_positions, rng=rng)
    search_strat = LocalSearch(pop_initializer, mutation_op, iterations=samples_per_iteration, rng=rng)
    return Algorithm(objfunc, search_strat, **kwargs)


def local_search_discrete(
    objfunc: ObjectiveFunc,
    resampled_components: int = 1,
    samples_per_iteration: int = 100,
    encoding: Optional[Encoding] = None,
    rng: Optional[RNGLike] = None,
    **kwargs,
) -> Algorithm:
    """Local Search for integer-coded vectors.

    Parameters
    ----------
    objfunc : ObjectiveFunc
        The objective function to optimize.
    resampled_components : int, optional
        Number of components resampled per mutation (default 1).
    samples_per_iteration : int, optional
        Number of samples evaluated per iteration (default 100).
    encoding : Encoding, optional
        Encoding applied to the genotype.
    rng : RNGLike, optional
        Random seed or generator.
    **kwargs
        Forwarded to :class:`Algorithm`.
    """

    rng = check_rng(rng)
    pop_initializer = UniformInitializer(
        objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=1, dtype=int, encoding=encoding, rng=rng
    )
    mutation_op = create_operator("random.reset", initializer=pop_initializer, n=resampled_components, rng=rng)
    search_strat = LocalSearch(pop_initializer, mutation_op, iterations=samples_per_iteration, rng=rng)
    return Algorithm(objfunc, search_strat, **kwargs)


def local_search_real(
    objfunc: ObjectiveFunc,
    mutation_strength: float = 1e-2,
    mutated_components: int = 1,
    samples_per_iteration: int = 100,
    encoding: Optional[Encoding] = None,
    rng: Optional[RNGLike] = None,
    **kwargs,
) -> Algorithm:
    """Local Search for real-coded vectors.

    Parameters
    ----------
    objfunc : ObjectiveFunc
        The objective function to optimize.
    mutation_strength : float, optional
        Standard deviation of Gaussian mutation (default 1e-2).
    mutated_components : int, optional
        Number of components mutated per individual (default 1).
    samples_per_iteration : int, optional
        Number of samples evaluated per iteration (default 100).
    encoding : Encoding, optional
        Encoding applied to the genotype.
    rng : RNGLike, optional
        Random seed or generator.
    **kwargs
        Forwarded to :class:`Algorithm`.
    """

    rng = check_rng(rng)
    pop_initializer = UniformInitializer(
        objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=1, dtype=float, encoding=encoding, rng=rng
    )
    mutation_op = create_operator("mutation.gaussian_mutation", F=mutation_strength, N=mutated_components, rng=rng)
    search_strat = LocalSearch(pop_initializer, mutation_op, iterations=samples_per_iteration, rng=rng)
    return Algorithm(objfunc, search_strat, **kwargs)
