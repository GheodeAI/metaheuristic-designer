"""
Univariate Marginal Distribution Algorithm (UMDA) strategies.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
from ...parent_selection_base import ParentSelection
from ...survivor_selection_base import SurvivorSelection
from ...initializer import Initializer
from ...operators import create_operator
from ...schedulable_parameter import SchedulableParameter
from ...utils import check_rng, RNGLike, VectorLike, ScalarLike
from ..eda_strategy import EDAStrategy


class BernoulliUMDA(EDAStrategy):
    """
    UMDA for binary vectors using a Bernoulli distribution.

    The probability vector is estimated from the selected parents
    (no smoothing).  Gaussian noise can optionally be added.

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
        Display name (default ``"BernoulliUMDA"``).
    offspring_size : int or SchedulableParameter, optional
        Number of offspring per generation.
    rng : RNGLike, optional
        Random number generator.
    p : float or array-like, optional
        Initial probability (default 0.5).
    noise : float, optional
        Gaussian noise standard deviation (default 0).
    **kwargs
        Forwarded to :class:`EDAStrategy`.
    """

    def __init__(
        self,
        initializer: Initializer,
        parent_sel: ParentSelection = None,
        survivor_sel: SurvivorSelection = None,
        name: str = "BernoulliUMDA",
        offspring_size: Optional[int | SchedulableParameter] = None,
        rng: Optional[RNGLike] = None,
        p: ScalarLike | SchedulableParameter = 0.5,
        noise: ScalarLike | SchedulableParameter = 0,
        **kwargs,
    ):
        rng = check_rng(rng)

        super().__init__(
            initializer,
            operator=create_operator("full_resampling", distribution="Bernoulli", p=np.asarray(p), rng=rng),
            parent_sel=parent_sel,
            survivor_sel=survivor_sel,
            offspring_size=offspring_size,
            name=name,
            rng=rng,
            # Forced kwargs
            noise=noise,
            **kwargs,
        )

        self.test_rng = np.random.default_rng(42)

    def estimate_parameters(self, population):
        old_p = self.operator.params.p

        population_matrix = population.genotype_matrix
        new_p = population_matrix.mean(axis=0)
        new_p += self.rng.normal(0, self.params.noise, size=old_p.shape)
        new_p = np.clip(new_p, 0, 1)

        self.operator.update_kwargs(p=new_p)

        return self.operator


class BinomialUMDA(EDAStrategy):
    """
    UMDA for discrete vectors using a Binomial distribution.

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
        Display name (default ``"BinomialUMDA"``).
    offspring_size : int or SchedulableParameter, optional
        Number of offspring per generation.
    rng : RNGLike, optional
        Random number generator.
    p : float or array-like, optional
        Initial success probability (default 0.5).
    n : int or array-like
        Number of trials. **Must be provided**; there is no default.
    noise : float, optional
        Gaussian noise standard deviation (default 0).
    **kwargs
        Forwarded to :class:`EDAStrategy`.
    """

    def __init__(
        self,
        initializer: Initializer,
        parent_sel: ParentSelection = None,
        survivor_sel: SurvivorSelection = None,
        name: str = "BinomialUMDA",
        offspring_size: Optional[int | SchedulableParameter] = None,
        rng: Optional[RNGLike] = None,
        p: ScalarLike | SchedulableParameter = 0.5,
        n: ScalarLike | SchedulableParameter = None,
        noise=0,
        **kwargs,
    ):
        rng = check_rng(rng)

        if n is None:
            raise ValueError("You must specify the value for the parameters `n`, usually it will be the number of possible categorical values.")

        super().__init__(
            initializer,
            operator=create_operator("full_resampling", distribution="Binomial", p=np.asarray(p), n=np.asarray(n), rng=rng),
            parent_sel=parent_sel,
            survivor_sel=survivor_sel,
            offspring_size=offspring_size,
            name=name,
            rng=rng,
            # Forced kwargs
            noise=noise,
            **kwargs,
        )

    def estimate_parameters(self, population):
        old_p = self.operator.params.p

        n = self.operator.params.n
        population_matrix = population.genotype_matrix
        new_p = population_matrix.sum(axis=0) / (n * population_matrix.shape[0])
        new_p += self.rng.normal(0, self.params.noise, size=old_p.shape)
        new_p = np.clip(new_p, 0, 1)

        self.operator.update_kwargs(p=new_p)

        return self.operator


class GaussianUMDA(EDAStrategy):
    """
    UMDA for continuous vectors using a Gaussian distribution.

    The location vector is estimated from the selected parents.
    Gaussian noise can optionally be added.

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
        Display name (default ``"GaussianUMDA"``).
    offspring_size : int or SchedulableParameter, optional
        Number of offspring per generation.
    rng : RNGLike, optional
        Random number generator.
    loc : float or array-like, optional
        Initial mean (default 0).
    scale : float or array-like, optional
        Standard deviation (default 1).
    noise : float, optional
        Gaussian noise standard deviation added to *loc* (default 0).
    **kwargs
        Forwarded to :class:`EDAStrategy`.
    """

    def __init__(
        self,
        initializer: Initializer,
        parent_sel: ParentSelection = None,
        survivor_sel: SurvivorSelection = None,
        name: str = "GaussianUMDA",
        offspring_size: Optional[int | SchedulableParameter] = None,
        rng=None,
        loc: ScalarLike | VectorLike | SchedulableParameter = 0,
        scale: ScalarLike | VectorLike | SchedulableParameter = 1,
        noise: ScalarLike | SchedulableParameter = 0,
        **kwargs,
    ):
        rng = check_rng(rng)

        super().__init__(
            initializer=initializer,
            operator=create_operator("full_resampling", distribution="gaussian", loc=np.asarray(loc), scale=np.asarray(scale), rng=rng),
            parent_sel=parent_sel,
            survivor_sel=survivor_sel,
            offspring_size=offspring_size,
            name=name,
            rng=rng,
            # Forced Kwargs
            noise=noise,
            **kwargs,
        )

    def estimate_parameters(self, population):
        old_loc = self.operator.params.loc

        population_matrix = population.genotype_matrix
        new_loc = population_matrix.mean(axis=0)
        new_loc += self.rng.normal(0, self.params.noise, size=old_loc.shape)

        self.operator.update_kwargs(loc=new_loc)

        return self.operator
