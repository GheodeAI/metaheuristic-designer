"""
Ready-to-run Bayesian optimization wrappers.
"""

from __future__ import annotations
from typing import Optional

from ..encoding import Encoding
from ..objective_function import ObjectiveFunc
from ..algorithm import Algorithm
from ..initializers import UniformInitializer
from ..strategies import BayesianOptimization
from ..utils import RNGLike, check_rng


def bayesian_optimization_binary(
    objfunc: ObjectiveFunc,
    population_size: int = 50,
    encoding: Optional[Encoding] = None,
    rng: Optional[RNGLike] = None,
    **kwargs,
) -> Algorithm:
    """
    Bayesian optimization for binary-coded vectors (not supported yet).
    """
    raise NotImplementedError("Bayesian optimization is only available for real-coded vectors.")


def bayesian_optimization_discrete(
    objfunc: ObjectiveFunc,
    population_size: int = 50,
    encoding: Optional[Encoding] = None,
    rng: Optional[RNGLike] = None,
    **kwargs,
) -> Algorithm:
    """
    Bayesian optimization for integer-coded vectors (not supported yet).
    """
    raise NotImplementedError("Bayesian optimization is only available for real-coded vectors.")


def bayesian_optimization_real(
    objfunc: ObjectiveFunc,
    population_size: int = 50,
    encoding: Optional[Encoding] = None,
    rng: Optional[RNGLike] = None,
    **kwargs,
) -> Algorithm:
    """Bayesian optimization for real-coded vectors.

    Parameters
    ----------
    objfunc : ObjectiveFunc
        The objective function to optimize.
    population_size : int, optional
        Number of individuals in the initial population (default 50).
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
    search_strategy = BayesianOptimization(
        initializer=pop_initializer,
        objfunc=objfunc,
        rng=rng,
    )
    return Algorithm(objfunc, search_strategy, **kwargs)
