"""
Ready-to-run Simulated Annealing wrappers.
"""

from __future__ import annotations
from typing import Optional
import numpy as np

from metaheuristic_designer.encoding import Encoding
from metaheuristic_designer.objective_function import ObjectiveFunc
from ..initializers import UniformInitializer, PermInitializer
from ..encodings import TypeCastEncoding
from ..strategies import SA
from ..algorithms import Algorithm
from ..operators import create_operator
from ..utils import RNGLike, check_rng


def simulated_annealing_binary(
    objfunc: ObjectiveFunc,
    mutated_bits: int = 1,
    initial_temperature: float = 1.0,
    alpha: float = 0.997,
    iterations: int = 100,
    encoding: Optional[Encoding] = None,
    rng: Optional[RNGLike] = None,
    **kwargs,
) -> Algorithm:
    """Simulated Annealing for binary-coded vectors.

    Parameters
    ----------
    objfunc : ObjectiveFunc
        The objective function to optimize.
    mutated_bits : int, optional
        Number of bits flipped per mutation (default 1).
    initial_temperature : float, optional
        Starting temperature (default 1.0).
    alpha : float, optional
        Cooling factor per iteration (default 0.997).
    iterations : int, optional
        Number of iterations at constant temperature (default 100).
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
    search_strat = SA(pop_initializer, mutation_op, temperature_init=initial_temperature, alpha=alpha, iterations=iterations, rng=rng)
    return Algorithm(objfunc, search_strat, **kwargs)


def simulated_annealing_permutation(
    objfunc: ObjectiveFunc,
    swapped_positions: int = 2,
    initial_temperature: float = 1.0,
    alpha: float = 0.997,
    iterations: int = 100,
    encoding: Optional[Encoding] = None,
    rng: Optional[RNGLike] = None,
    **kwargs,
) -> Algorithm:
    """Simulated Annealing for permutation-coded vectors.

    Parameters
    ----------
    objfunc : ObjectiveFunc
        The objective function to optimize.
    swapped_positions : int, optional
        Number of positions swapped per mutation (default 2).
    initial_temperature : float, optional
        Starting temperature (default 1.0).
    alpha : float, optional
        Cooling factor per iteration (default 0.997).
    iterations : int, optional
        Number of iterations at constant temperature (default 100).
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
    search_strat = SA(pop_initializer, mutation_op, temperature_init=initial_temperature, alpha=alpha, iterations=iterations, rng=rng)
    return Algorithm(objfunc, search_strat, **kwargs)


def simulated_annealing_discrete(
    objfunc: ObjectiveFunc,
    resampled_components: int = 1,
    initial_temperature: float = 1.0,
    alpha: float = 0.997,
    iterations: int = 100,
    encoding: Optional[Encoding] = None,
    rng: Optional[RNGLike] = None,
    **kwargs,
) -> Algorithm:
    """Simulated Annealing for integer-coded vectors.

    Parameters
    ----------
    objfunc : ObjectiveFunc
        The objective function to optimize.
    resampled_components : int, optional
        Number of components resampled per mutation (default 1).
    initial_temperature : float, optional
        Starting temperature (default 1.0).
    alpha : float, optional
        Cooling factor per iteration (default 0.997).
    iterations : int, optional
        Number of iterations at constant temperature (default 100).
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
    mutation_op = create_operator("random.reset", n=resampled_components, rng=rng)
    search_strat = SA(pop_initializer, mutation_op, temperature_init=initial_temperature, alpha=alpha, iterations=iterations, rng=rng)
    return Algorithm(objfunc, search_strat, **kwargs)


def simulated_annealing_real(
    objfunc: ObjectiveFunc,
    mutation_strength: float = 1e-2,
    mutated_components: int = 1,
    initial_temperature: float = 1.0,
    alpha: float = 0.997,
    iterations: int = 100,
    encoding: Optional[Encoding] = None,
    rng: Optional[RNGLike] = None,
    **kwargs,
) -> Algorithm:
    """Simulated Annealing for real-coded vectors.

    Parameters
    ----------
    objfunc : ObjectiveFunc
        The objective function to optimize.
    mutation_strength : float, optional
        Standard deviation of Gaussian mutation (default 1e-2).
    mutated_components : int, optional
        Number of components mutated per individual (default 1).
    initial_temperature : float, optional
        Starting temperature (default 1.0).
    alpha : float, optional
        Cooling factor per iteration (default 0.997).
    iterations : int, optional
        Number of iterations at constant temperature (default 100).
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
    search_strat = SA(pop_initializer, mutation_op, temperature_init=initial_temperature, alpha=alpha, iterations=iterations, rng=rng)
    return Algorithm(objfunc, search_strat, **kwargs)
