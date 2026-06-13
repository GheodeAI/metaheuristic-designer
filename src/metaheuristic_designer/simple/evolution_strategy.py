"""
Ready-to-run Evolution Strategy wrappers.
"""

from __future__ import annotations
from typing import Optional
import numpy as np

from ..encoding import Encoding
from ..objective_function import ObjectiveFunc
from ..initializers import UniformInitializer, PermInitializer
from ..encodings import TypeCastEncoding
from ..strategies import ES
from ..algorithms import Algorithm
from ..operators import create_operator
from ..survivor_selection import create_survivor_selection
from ..utils import RNGLike, check_rng


def evolution_strategy_binary(
    objfunc: ObjectiveFunc,
    mutated_bits: int = 1,
    population_size: int = 100,
    offspring_size: int = 500,
    elitist: bool = False,
    encoding: Optional[Encoding] = None,
    rng: Optional[RNGLike] = None,
    **kwargs,
) -> Algorithm:
    """Evolution Strategy for binary-coded vectors.

    Parameters
    ----------
    objfunc : ObjectiveFunc
        The objective function to optimize.
    mutated_bits : int, optional
        Number of bits flipped per mutation (default 1).
    population_size : int, optional
        Population size (default 100).
    offspring_size : int, optional
        Number of offspring per generation (default 500).
    elitist : bool, optional
        If ``True``, use (μ+λ) selection; otherwise (μ,λ).
    encoding : Encoding, optional
        Encoding; defaults to :class:`TypeCastEncoding` (int → bool).
    rng : RNGLike, optional
        Random seed or generator.
    \\*\\*kwargs
        Forwarded to :class:`Algorithm`.
    """

    rng = check_rng(rng)
    encoding = TypeCastEncoding(int, bool) if encoding is None else encoding
    pop_initializer = UniformInitializer(objfunc.dimension, 0, 1, population_size=population_size, dtype=np.uint8, encoding=encoding, rng=rng)
    mutation_op = create_operator("mutation.bitflip", N=mutated_bits, rng=rng)
    method = "keep_best" if elitist else "keep_offspring"
    survivor_sel = create_survivor_selection(method, rng=rng)
    search_strat = ES(initializer=pop_initializer, mutation_op=mutation_op, survivor_sel=survivor_sel, offspring_size=offspring_size, rng=rng)
    return Algorithm(objfunc, search_strat, **kwargs)


def evolution_strategy_permutation(
    objfunc: ObjectiveFunc,
    swapped_positions: int = 2,
    population_size: int = 100,
    offspring_size: int = 500,
    elitist: bool = False,
    encoding: Optional[Encoding] = None,
    rng: Optional[RNGLike] = None,
    **kwargs,
) -> Algorithm:
    """Evolution Strategy for permutation-coded vectors.

    Parameters
    ----------
    objfunc : ObjectiveFunc
        The objective function to optimize.
    swapped_positions : int, optional
        Number of positions swapped per mutation (default 2).
    population_size : int, optional
        Population size (default 100).
    offspring_size : int, optional
        Number of offspring per generation (default 500).
    elitist : bool, optional
        If ``True``, use (μ+λ) selection; otherwise (μ,λ).
    encoding : Encoding, optional
        Encoding applied to the genotype.
    rng : RNGLike, optional
        Random seed or generator.
    \\*\\*kwargs
        Forwarded to :class:`Algorithm`.
    """

    rng = check_rng(rng)
    pop_initializer = PermInitializer(objfunc.dimension, population_size=population_size, encoding=encoding, rng=rng)
    mutation_op = create_operator("permutation.swap", N=swapped_positions, rng=rng)
    method = "keep_best" if elitist else "keep_offspring"
    survivor_sel = create_survivor_selection(method, rng=rng)
    search_strat = ES(initializer=pop_initializer, mutation_op=mutation_op, survivor_sel=survivor_sel, offspring_size=offspring_size, rng=rng)
    return Algorithm(objfunc, search_strat, **kwargs)


def evolution_strategy_discrete(
    objfunc: ObjectiveFunc,
    resampled_components: int = 1,
    population_size: int = 100,
    offspring_size: int = 500,
    elitist: bool = False,
    encoding: Optional[Encoding] = None,
    rng: Optional[RNGLike] = None,
    **kwargs,
) -> Algorithm:
    """Evolution Strategy for integer-coded vectors.

    Parameters
    ----------
    objfunc : ObjectiveFunc
        The objective function to optimize.
    resampled_components : int, optional
        Number of components resampled per mutation (default 1).
    population_size : int, optional
        Population size (default 100).
    offspring_size : int, optional
        Number of offspring per generation (default 500).
    elitist : bool, optional
        If ``True``, use (μ+λ) selection; otherwise (μ,λ).
    encoding : Encoding, optional
        Encoding applied to the genotype.
    rng : RNGLike, optional
        Random seed or generator.
    \\*\\*kwargs
        Forwarded to :class:`Algorithm`.
    """

    rng = check_rng(rng)
    pop_initializer = UniformInitializer(
        objfunc.dimension,
        objfunc.lower_bound,
        objfunc.upper_bound,
        population_size=population_size,
        dtype=int,
        encoding=encoding,
        rng=rng,
    )
    mutation_op = create_operator("random.reset", initializer=pop_initializer, n=resampled_components, rng=rng)
    method = "keep_best" if elitist else "keep_offspring"
    survivor_sel = create_survivor_selection(method, rng=rng)
    search_strat = ES(initializer=pop_initializer, mutation_op=mutation_op, survivor_sel=survivor_sel, offspring_size=offspring_size, rng=rng)
    return Algorithm(objfunc, search_strat, **kwargs)


def evolution_strategy_real(
    objfunc: ObjectiveFunc,
    mutation_strength: float = 1e-2,
    mutated_components: int = 1,
    population_size: int = 100,
    offspring_size: int = 500,
    elitist: bool = False,
    encoding: Optional[Encoding] = None,
    rng: Optional[RNGLike] = None,
    **kwargs,
) -> Algorithm:
    """Evolution Strategy for integer-coded vectors.

    Parameters
    ----------
    objfunc : ObjectiveFunc
        The objective function to optimize.
    resampled_components : int, optional
        Number of components resampled per mutation (default 1).
    population_size : int, optional
        Population size (default 100).
    offspring_size : int, optional
        Number of offspring per generation (default 500).
    elitist : bool, optional
        If ``True``, use (μ+λ) selection; otherwise (μ,λ).
    encoding : Encoding, optional
        Encoding applied to the genotype.
    rng : RNGLike, optional
        Random seed or generator.
    \\*\\*kwargs
        Forwarded to :class:`Algorithm`.
    """

    rng = check_rng(rng)
    pop_initializer = UniformInitializer(
        objfunc.dimension,
        objfunc.lower_bound,
        objfunc.upper_bound,
        population_size=population_size,
        dtype=float,
        encoding=encoding,
        rng=rng,
    )
    mutation_op = create_operator("mutation.gaussian_mutation", F=mutation_strength, N=mutated_components, rng=rng)
    method = "keep_best" if elitist else "keep_offspring"
    survivor_sel = create_survivor_selection(method, rng=rng)
    search_strat = ES(initializer=pop_initializer, mutation_op=mutation_op, survivor_sel=survivor_sel, offspring_size=offspring_size, rng=rng)
    return Algorithm(objfunc, search_strat, **kwargs)
