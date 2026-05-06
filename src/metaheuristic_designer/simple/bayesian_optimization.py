from __future__ import annotations
from ..algorithm import Algorithm
from ..initializers import UniformInitializer
from ..strategies import BayesianOptimization
from ..utils import check_random_state


def bayesian_optimization_binary(
    objfunc,
    population_size: int = 50,
    acquisition_function: str = "EI",
    encoding=None,
    random_state=None,
    **kwargs,
):
    """
    Bayesian Optimisation for binary-coded vectors (not supported yet).
    """
    raise NotImplementedError("Bayesian Optimisation is only available for real-coded vectors.")


def bayesian_optimization_discrete(
    objfunc,
    population_size: int = 50,
    acquisition_function: str = "EI",
    encoding=None,
    random_state=None,
    **kwargs,
):
    """
    Bayesian Optimisation for integer-coded vectors (not supported yet).
    """
    raise NotImplementedError("Bayesian Optimisation is only available for real-coded vectors.")


def bayesian_optimization_real(
    objfunc,
    population_size: int = 50,
    encoding=None,
    random_state=None,
    **kwargs,
):
    """
    Bayesian Optimisation for real-coded vectors.
    """
    random_state = check_random_state(random_state)
    pop_initializer = UniformInitializer(
        objfunc.dimension,
        objfunc.lower_bound,
        objfunc.upper_bound,
        pop_size=population_size,
        dtype=float,
        encoding=encoding,
        random_state=random_state,
    )
    search_strategy = BayesianOptimization(
        initializer=pop_initializer,
        random_state=random_state,
    )
    return Algorithm(objfunc, search_strategy, **kwargs)
