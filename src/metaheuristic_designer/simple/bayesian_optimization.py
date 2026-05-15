"""
Ready-to-run Bayesian Optimisation wrappers.
"""

from __future__ import annotations
from typing import Optional

from ..encoding import Encoding
from ..objective_function import ObjectiveFunc
from ..algorithm import Algorithm
from ..initializers import UniformInitializer
from ..strategies import BayesianOptimization
from ..utils import RNGLike, check_random_state


def bayesian_optimization_binary(
    objfunc: ObjectiveFunc,
    population_size: int = 50,
    encoding: Optional[Encoding] = None,
    random_state: Optional[RNGLike] = None,
    **kwargs,
) -> Algorithm:
    """
    Bayesian Optimisation for binary-coded vectors (not supported yet).
    """
    raise NotImplementedError("Bayesian Optimisation is only available for real-coded vectors.")


def bayesian_optimization_discrete(
    objfunc: ObjectiveFunc,
    population_size: int = 50,
    encoding: Optional[Encoding] = None,
    random_state: Optional[RNGLike] = None,
    **kwargs,
) -> Algorithm:
    """
    Bayesian Optimisation for integer-coded vectors (not supported yet).
    """
    raise NotImplementedError("Bayesian Optimisation is only available for real-coded vectors.")


def bayesian_optimization_real(
    objfunc: ObjectiveFunc,
    population_size: int = 50,
    encoding: Optional[Encoding] = None,
    random_state: Optional[RNGLike] = None,
    **kwargs,
) -> Algorithm:
    """Bayesian Optimisation for real-coded vectors.

    Parameters
    ----------
    objfunc : ObjectiveFunc
        The objective function to optimise.
    population_size : int, optional
        Number of individuals in the initial population (default 50).
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
    search_strategy = BayesianOptimization(
        initializer=pop_initializer,
        random_state=random_state,
    )
    return Algorithm(objfunc, search_strategy, **kwargs)
