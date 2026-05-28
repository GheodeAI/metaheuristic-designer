"""
Ready-to-run Differential Evolution wrappers.
"""

from __future__ import annotations
from typing import Optional

from ..objective_function import ObjectiveFunc
from ..encoding import Encoding
from ..algorithm import Algorithm
from ..initializers import UniformInitializer
from ..encodings import TypeCastEncoding, SigmoidEncoding
from ..strategies import DE
from ..utils import RNGLike, check_random_state


def differential_evolution_binary(
    objfunc: ObjectiveFunc,
    population_size: int = 100,
    F: float = 0.8,
    Cr: float = 0.9,
    de_operator_name: str = "de/rand/1",
    encoding: Optional[Encoding] = None,
    random_state: Optional[RNGLike] = None,
    **kwargs,
) -> Algorithm:
    """Differential Evolution for binary-encoded vectors.

    Parameters
    ----------
    objfunc : ObjectiveFunc
        The objective function to optimize.
    population_size : int, optional
        Population size (default 100).
    F : float, optional
        Mutation scale factor (default 0.8).
    Cr : float, optional
        Crossover probability (default 0.9).
    de_operator_name : str, optional
        DE variant (default ``"de/rand/1"``).
    encoding : Encoding, optional
        Encoding; defaults to :class:`SigmoidEncoding`.
    random_state : RNGLike, optional
        Random seed or generator.
    **kwargs
        Forwarded to :class:`Algorithm`.
    """

    random_state = check_random_state(random_state)
    if encoding is None:
        encoding = SigmoidEncoding(as_probability=False, threshold=0.5)
    pop_initializer = UniformInitializer(
        objfunc.dimension,
        objfunc.lower_bound,
        objfunc.upper_bound,
        population_size=population_size,
        dtype=float,
        encoding=encoding,
        random_state=random_state,
    )
    search_strat = DE(
        initializer=pop_initializer,
        de_operator_name=de_operator_name,
        F=F,
        Cr=Cr,
        random_state=random_state,
    )
    return Algorithm(objfunc, search_strat, **kwargs)


def differential_evolution_discrete(
    objfunc: ObjectiveFunc,
    population_size: int = 100,
    F: float = 0.8,
    Cr: float = 0.9,
    de_operator_name: str = "de/rand/1",
    encoding: Optional[Encoding] = None,
    random_state: Optional[RNGLike] = None,
    **kwargs,
) -> Algorithm:
    """Differential Evolution for integer-coded vectors.

    Parameters
    ----------
    objfunc : ObjectiveFunc
        The objective function to optimize.
    population_size : int, optional
        Population size (default 100).
    F : float, optional
        Mutation scale factor (default 0.8).
    Cr : float, optional
        Crossover probability (default 0.9).
    de_operator_name : str, optional
        DE variant (default ``"de/rand/1"``).
    encoding : Encoding, optional
        Encoding; defaults to :class:`TypeCastEncoding` (float → int).
    random_state : RNGLike, optional
        Random seed or generator.
    **kwargs
        Forwarded to :class:`Algorithm`.
    """

    random_state = check_random_state(random_state)
    if encoding is None:
        encoding = TypeCastEncoding(float, int)
    pop_initializer = UniformInitializer(
        objfunc.dimension,
        objfunc.lower_bound,
        objfunc.upper_bound,
        population_size=population_size,
        dtype=float,
        encoding=encoding,
        random_state=random_state,
    )
    search_strat = DE(
        initializer=pop_initializer,
        de_operator_name=de_operator_name,
        F=F,
        Cr=Cr,
        random_state=random_state,
    )
    return Algorithm(objfunc, search_strat, **kwargs)


def differential_evolution_real(
    objfunc: ObjectiveFunc,
    population_size: int = 100,
    F: float = 0.8,
    Cr: float = 0.9,
    de_operator_name: str = "de/rand/1",
    encoding: Optional[Encoding] = None,
    random_state: Optional[RNGLike] = None,
    **kwargs,
) -> Algorithm:
    """Differential Evolution for real-coded vectors.

    Parameters
    ----------
    objfunc : ObjectiveFunc
        The objective function to optimizes.
    population_size : int, optional
        Population size (default 100).
    F : float, optional
        Mutation scale factor (default 0.8).
    Cr : float, optional
        Crossover probability (default 0.9).
    de_operator_name : str, optional
        DE variant (default ``"de/rand/1"``).
    encoding : Encoding, optional
        Encoding applied to the genotype.
    random_state : RNGLike, optional
        Random seed or generator.
    **kwargs
        Forwarded to :class:`Algorithm`.
    """

    random_state = check_random_state(random_state)
    pop_initializer = UniformInitializer(
        objfunc.dimension,
        objfunc.lower_bound,
        objfunc.upper_bound,
        population_size=population_size,
        dtype=float,
        encoding=encoding,
        random_state=random_state,
    )
    search_strat = DE(
        initializer=pop_initializer,
        de_operator_name=de_operator_name,
        F=F,
        Cr=Cr,
        random_state=random_state,
    )
    return Algorithm(objfunc, search_strat, **kwargs)
