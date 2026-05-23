"""
Population-Based Incremental Learning (PBIL) strategies.
"""

from __future__ import annotations
from typing import Optional
import numpy as np

from ...population import Population
from ...operators import create_operator, Operator
from ...parent_selection_base import ParentSelection
from ...survivor_selection_base import SurvivorSelection
from ...initializer import Initializer
from ...schedulable_parameter import SchedulableParameter
from ..eda_strategy import EDAStrategy
from ...utils import check_random_state


class BernoulliPBIL(EDAStrategy):
    """
    PBIL for binary vectors using a Bernoulli distribution.

    The probability vector *p* is updated each generation with a
    learning rate and optional Gaussian noise, then a new population
    is sampled.

    Reference: https://doi.org/10.1016/j.swevo.2011.08.003

    Parameters
    ----------
    initializer : Initializer
        Population initializer.
    parent_sel : ParentSelection, optional
        Parent selection method.
    survivor_sel : SurvivorSelection, optional
        Survivor selection method.
    name : str, optional
        Display name (default ``"BernoulliPBIL"``).
    offspring_size : int or SchedulableParameter, optional
        Number of offspring per generation.
    random_state : RNGLike, optional
        Random number generator.
    p : array-like, optional
        Initial probability vector.  Defaults to uniform over [0,1].
    lr : float, optional
        Learning rate for updating *p* (default 1e-3).
    noise : float, optional
        Standard deviation of Gaussian noise added to *p* (default 0).
    **kwargs
        Forwarded to :class:`EDAStrategy
    `.
    """

    def __init__(
        self,
        initializer: Initializer,
        parent_sel: ParentSelection = None,
        survivor_sel: SurvivorSelection = None,
        name: str = "BernoulliPBIL",
        offspring_size: Optional[int | SchedulableParameter] = None,
        random_state=None,
        p=None,
        lr=1e-3,
        noise=0,
        **kwargs,
    ):
        random_state = check_random_state(random_state)

        super().__init__(
            initializer,
            operator=create_operator("full_resampling", distribution="bernoulli", p=p, random_state=random_state),
            parent_sel=parent_sel,
            survivor_sel=survivor_sel,
            offspring_size=offspring_size,
            name=name,
            random_state=random_state,
            # Forced kwargs
            lr=lr,
            noise=noise,
            **kwargs,
        )

    def estimate_parameters(self, population):
        old_p = self.operator.params.p

        population_matrix = population.genotype_matrix
        new_p = population_matrix.mean(axis=0)
        if old_p is not None:
            new_p = (1 - self.params.lr) * old_p + self.params.lr * new_p
            new_p += self.random_state.normal(0, self.params.noise, size=np.asarray(old_p).shape)
        new_p = np.clip(new_p, 0, 1)

        self.operator.update_kwargs(p=new_p)

        return self.operator


class BinomialPBIL(EDAStrategy):
    """
    PBIL for discrete vectors using a Binomial distribution.

    Reference: https://doi.org/10.1016/j.swevo.2011.08.003

    Parameters
    ----------
    initializer : Initializer
        Population initializer.
    parent_sel : ParentSelection, optional
        Parent selection method.
    survivor_sel : SurvivorSelection, optional
        Survivor selection method.
    name : str, optional
        Display name (default ``"BinomialPBIL"``).
    offspring_size : int or SchedulableParameter, optional
        Number of offspring per generation.
    random_state : RNGLike, optional
        Random number generator.
    p : float or array-like, optional
        Initial success probability (default 0.5).
    n : int or array-like
        Number of trials. **Must be provided**; there is no default.
    lr : float, optional
        Learning rate (default 1e-3).
    noise : float, optional
        Gaussian noise standard deviation (default 0).
    **kwargs
        Forwarded to :class:`EDAStrategy
    `.
    """

    def __init__(
        self,
        initializer: Initializer,
        parent_sel: ParentSelection = None,
        survivor_sel: SurvivorSelection = None,
        name: str = "BernoulliPBIL",
        offspring_size: Optional[int | SchedulableParameter] = None,
        random_state=None,
        p=0.5,
        n=None,
        lr=1e-3,
        noise=0,
        **kwargs,
    ):
        random_state = check_random_state(random_state)

        if n is None:
            raise ValueError("You must specify the value for the parameters `n`, usually it will be the number of possible categorical values.")

        super().__init__(
            initializer,
            operator=create_operator("full_resampling", distribution="Binomial", p=np.asarray(p), n=np.asarray(n), random_state=random_state),
            parent_sel=parent_sel,
            survivor_sel=survivor_sel,
            offspring_size=offspring_size,
            name=name,
            random_state=random_state,
            # Forced kwargs
            noise=noise,
            lr=lr,
            **kwargs,
        )

    def estimate_parameters(self, population: Population) -> Operator:
        n = self.operator.params.n
        old_p = self.operator.params.p

        population_matrix = population.genotype_matrix
        new_p = population_matrix.sum(axis=0) / (n * population_matrix.shape[0])

        if old_p is not None:
            new_p = (1 - self.params.lr) * old_p + self.params.lr * new_p
            new_p += self.random_state.normal(0, self.params.noise, size=old_p.shape)
            new_p = np.clip(new_p, 0, 1)

        self.operator.update_kwargs(p=new_p)

        return self.operator


class GaussianPBIL(EDAStrategy):
    """
    PBIL for continuous vectors using a Gaussian distribution.

    The location vector *loc* is updated each generation with a
    learning rate and optional Gaussian noise, then a new population
    is sampled.

    Reference: https://doi.org/10.1016/j.swevo.2011.08.003

    Parameters
    ----------
    initializer : Initializer
        Population initializer.
    parent_sel : ParentSelection, optional
        Parent selection method.
    survivor_sel : SurvivorSelection, optional
        Survivor selection method.
    name : str, optional
        Display name (default ``"GaussianPBIL"``).
    offspring_size : int or SchedulableParameter, optional
        Number of offspring per generation.
    random_state : RNGLike, optional
        Random number generator.
    loc : array-like, optional
        Initial mean vector (default ``None``; the operator uses a
        fallback).
    scale : float or array-like, optional
        Standard deviation (default 1).
    lr : float, optional
        Learning rate (default 1e-3).
    noise : float, optional
        Gaussian noise standard deviation added to *loc* (default 0).
    **kwargs
        Forwarded to :class:`EDAStrategy
    `.
    """

    def __init__(
        self,
        initializer: Initializer,
        parent_sel: ParentSelection = None,
        survivor_sel: SurvivorSelection = None,
        name: str = "GaussianPBIL",
        offspring_size: Optional[int | SchedulableParameter] = None,
        random_state=None,
        loc=None,
        scale=1,
        lr=1e-3,
        noise=0,
        **kwargs,
    ):
        random_state = check_random_state(random_state)

        super().__init__(
            initializer,
            operator=create_operator("full_resampling", distribution="gaussian", loc=loc, scale=np.asarray(scale), random_state=random_state),
            parent_sel=parent_sel,
            survivor_sel=survivor_sel,
            offspring_size=offspring_size,
            name=name,
            random_state=random_state,
            # Forced kwargs
            lr=lr,
            noise=noise,
            **kwargs,
        )

    def estimate_parameters(self, population):
        old_loc = self.operator.params.loc

        population_matrix = population.genotype_matrix
        new_loc = population_matrix.mean(axis=0)

        if old_loc is not None:
            new_loc = (1 - self.params.lr) * old_loc + self.params.lr * new_loc
            new_loc += self.random_state.normal(0, self.params.noise, size=old_loc.shape)

        self.operator.update_kwargs(loc=new_loc)

        return self.operator
