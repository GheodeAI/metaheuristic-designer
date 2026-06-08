"""Ready-to-run Particle Swarm Optimization wrappers."""

from __future__ import annotations
from typing import Optional

from metaheuristic_designer.encoding import Encoding
from metaheuristic_designer.objective_function import ObjectiveFunc
from ..algorithm import Algorithm
from ..initializers import UniformInitializer
from ..encodings import TypeCastEncoding, SigmoidEncoding
from ..strategies import PSO
from ..utils import RNGLike, check_rng


def particle_swarm_binary(
    objfunc: ObjectiveFunc,
    population_size: int = 100,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    encoding: Optional[Encoding] = None,
    rng: Optional[RNGLike] = None,
    **kwargs,
) -> Algorithm:
    """Particle Swarm optimization for binary-coded vectors.

    Parameters
    ----------
    objfunc : ObjectiveFunc
        The objective function to optimize.
    population_size : int, optional
        Swarm size (default 100).
    w : float, optional
        Inertia weight (default 0.7).
    c1 : float, optional
        Cognitive acceleration coefficient (default 1.5).
    c2 : float, optional
        Social acceleration coefficient (default 1.5).
    encoding : Encoding, optional
        Encoding; defaults to :class:`SigmoidEncoding`.
    rng : RNGLike, optional
        Random seed or generator.
    **kwargs
        Forwarded to :class:`Algorithm`.
    """

    rng = check_rng(rng)
    if encoding is None:
        encoding = SigmoidEncoding(as_probability=False, threshold=0.5)
    pop_initializer = UniformInitializer(
        objfunc.dimension,
        objfunc.lower_bound,
        objfunc.upper_bound,
        population_size=population_size,
        dtype=float,
        encoding=encoding,
        rng=rng,
    )
    search_strat = PSO(
        initializer=pop_initializer,
        w=w,
        c1=c1,
        c2=c2,
        rng=rng,
    )
    return Algorithm(objfunc, search_strat, **kwargs)


def particle_swarm_discrete(
    objfunc: ObjectiveFunc,
    population_size: int = 100,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    encoding: Optional[Encoding] = None,
    rng: Optional[RNGLike] = None,
    **kwargs,
) -> Algorithm:
    """Particle Swarm optimization for integer-coded vectors.

    Parameters
    ----------
    objfunc : ObjectiveFunc
        The objective function to optimize.
    population_size : int, optional
        Swarm size (default 100).
    w : float, optional
        Inertia weight (default 0.7).
    c1 : float, optional
        Cognitive acceleration coefficient (default 1.5).
    c2 : float, optional
        Social acceleration coefficient (default 1.5).
    encoding : Encoding, optional
        Encoding; defaults to :class:`TypeCastEncoding` (float → int).
    rng : RNGLike, optional
        Random seed or generator.
    **kwargs
        Forwarded to :class:`Algorithm`.
    """
    rng = check_rng(rng)
    if encoding is None:
        encoding = TypeCastEncoding(float, int)
    pop_initializer = UniformInitializer(
        objfunc.dimension,
        objfunc.lower_bound,
        objfunc.upper_bound,
        population_size=population_size,
        dtype=float,
        encoding=encoding,
        rng=rng,
    )
    search_strat = PSO(
        initializer=pop_initializer,
        w=w,
        c1=c1,
        c2=c2,
        rng=rng,
    )
    return Algorithm(objfunc, search_strat, **kwargs)


def particle_swarm_real(
    objfunc: ObjectiveFunc,
    population_size: int = 100,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    encoding: Optional[Encoding] = None,
    rng: Optional[RNGLike] = None,
    **kwargs,
) -> Algorithm:
    """Particle Swarm Optimization for real-coded vectors.

    Parameters
    ----------
    objfunc : ObjectiveFunc
        The objective function to optimize.
    population_size : int, optional
        Swarm size (default 100).
    w : float, optional
        Inertia weight (default 0.7).
    c1 : float, optional
        Cognitive acceleration coefficient (default 1.5).
    c2 : float, optional
        Social acceleration coefficient (default 1.5).
    encoding : Encoding, optional
        Encoding applied to the genotype.
    rng : RNGLike, optional
        Random seed or generator.
    **kwargs
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
    search_strat = PSO(
        initializer=pop_initializer,
        w=w,
        c1=c1,
        c2=c2,
        rng=rng,
    )
    return Algorithm(objfunc, search_strat, **kwargs)
