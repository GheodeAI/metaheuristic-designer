from __future__ import annotations
from typing import Optional
import numpy as np
from ...parent_selection_base import ParentSelection
from ...survivor_selection_base import SurvivorSelection
from ...initializer import Initializer
from ..variable_population import VariablePopulation
from ...operators import create_operator
from ...schedulable_parameter import SchedulableParameter
from ...utils import check_random_state, RNGLike, VectorLike, ScalarLike


class BernoulliUMDA(VariablePopulation):
    """
    Estimation of distribution algorithm for binary vectors.
    https://doi.org/10.1016/j.swevo.2011.08.003
    """

    def __init__(
        self,
        initializer: Initializer,
        parent_sel: ParentSelection = None,
        survivor_sel: SurvivorSelection = None,
        name: str = "BernoulliUMDA",
        offspring_size: Optional[int | SchedulableParameter] = None,
        random_state=None,
        p: ScalarLike | SchedulableParameter = 0.5,
        noise: ScalarLike | SchedulableParameter = 0,
        **kwargs,
    ):
        self.random_state = check_random_state(random_state)

        super().__init__(
            initializer,
            operator=create_operator("full_resampling", distribution="Bernoulli", p=np.asarray(p), random_state=random_state),
            parent_sel=parent_sel,
            survivor_sel=survivor_sel,
            offspring_size=offspring_size,
            name=name,
            # Forced kwargs
            noise=noise,
            **kwargs,
        )

    def _batch_fit(self, population):
        population_matrix = population.genotype_matrix
        p_hat = population_matrix.mean(axis=0)

        return p_hat

    def perturb(self, parents, **kwargs):
        old_p = self.operator.params.p

        new_p = self._batch_fit(parents)
        new_p += self.random_state.normal(0, self.params.noise, size=old_p.shape)
        new_p = np.clip(new_p, 0, 1)

        self.operator.update_kwargs(p=new_p)

        return super().perturb(parents, **kwargs)


class BinomialUMDA(VariablePopulation):
    """
    Estimation of distribution algorithm for binary vectors.
    https://doi.org/10.1016/j.swevo.2011.08.003
    """

    def __init__(
        self,
        initializer: Initializer,
        parent_sel: ParentSelection = None,
        survivor_sel: SurvivorSelection = None,
        name: str = "BinomialUMDA",
        offspring_size: Optional[int | SchedulableParameter] = None,
        random_state: Optional[RNGLike] = None,
        p: ScalarLike | SchedulableParameter = 0.5,
        n: ScalarLike | SchedulableParameter = None,
        noise=0,
        **kwargs,
    ):
        self.random_state = check_random_state(random_state)

        if n is None:
            raise ValueError("You must specify the value for the parameters `n`, usually it will be the number of possible categorical values.")

        super().__init__(
            initializer,
            operator=create_operator("full_resampling", distribution="Binomial", p=np.asarray(p), n=np.asarray(n), random_state=random_state),
            parent_sel=parent_sel,
            survivor_sel=survivor_sel,
            offspring_size=offspring_size,
            name=name,
            # Forced kwargs
            noise=noise,
            **kwargs,
        )

    def _batch_fit(self, population):
        n = self.operator.params.n
        population_matrix = population.genotype_matrix
        p_hat = population_matrix.sum(axis=0) / (n * population_matrix.shape[0])

        return p_hat

    def perturb(self, parents, **kwargs):
        old_p = self.operator.params.p

        new_p = self._batch_fit(parents)
        new_p += self.random_state.normal(0, self.params.noise, size=old_p.shape)
        new_p = np.clip(new_p, 0, 1)

        self.operator.update_kwargs(p=new_p)

        return super().perturb(parents, **kwargs)


class GaussianUMDA(VariablePopulation):
    """
    Estimation of distribution algorithm for binary vectors.
    https://doi.org/10.1016/j.swevo.2011.08.003
    """

    def __init__(
        self,
        initializer: Initializer,
        parent_sel: ParentSelection = None,
        survivor_sel: SurvivorSelection = None,
        name: str = "GaussianUMDA",
        offspring_size: Optional[int | SchedulableParameter] = None,
        random_state=None,
        loc: ScalarLike | VectorLike | SchedulableParameter = 0,
        scale: ScalarLike | VectorLike | SchedulableParameter = 1,
        noise: ScalarLike | SchedulableParameter = 0,
        **kwargs,
    ):
        self.random_state = check_random_state(random_state)

        super().__init__(
            initializer=initializer,
            operator=create_operator("full_resampling", distribution="gaussian", loc=np.asarray(loc), scale=np.asarray(scale), random_state=random_state),
            parent_sel=parent_sel,
            survivor_sel=survivor_sel,
            offspring_size=offspring_size,
            name=name,
            # Forced Kwargs
            noise=noise,
            **kwargs,
        )

    def _batch_fit(self, population):
        population_matrix = population.genotype_matrix
        loc_hat = population_matrix.mean(axis=0)

        return loc_hat

    def perturb(self, parents, **kwargs):
        old_loc = self.operator.params.loc

        new_loc = self._batch_fit(parents)
        new_loc += self.random_state.normal(0, self.params.noise, size=old_loc.shape)

        self.operator.update_kwargs(loc=new_loc)

        return super().perturb(parents, **kwargs)
