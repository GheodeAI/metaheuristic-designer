from __future__ import annotations
from typing import Optional
import logging
import numpy as np

from metaheuristic_designer.parent_selection.parent_selection import create_parent_selection
from metaheuristic_designer.population import Population
from ...initializer import Initializer
from ...schedulable_parameter import SchedulableParameter
from ...survivor_selection_base import SurvivorSelection
from ...parent_selection_base import ParentSelection
from ..variable_population import VariablePopulation
from ...operators import create_operator
from ...utils import VectorLike, check_random_state

logger = logging.getLogger(__name__)


class CMA_ES(VariablePopulation):
    """
    CMA-ES algorithm.
    """

    def __init__(
        self,
        initializer: Initializer,
        survivor_sel: SurvivorSelection = None,
        name: str = "CMA-ES",
        offspring_size: Optional[int | SchedulableParameter] = None,
        random_state=None,
        mean: Optional[VectorLike] = None,
        sigma: Optional[VectorLike] = None,
        **kwargs,
    ):
        self.random_state = check_random_state(random_state)

        logger.info(
            "In CMA-ES the initializer does not generate solutions, it merely indicates the population size and encoding. Don't expect different results from changing the initializer."
        )
        self._b_matrix = None
        self._d_diag = None

        super().__init__(
            initializer,
            operator=create_operator("mutation.full_resampling", distrib="multivariate_normal", mean=None, cov=None),
            parent_sel=create_parent_selection("best", amount=initializer.pop_size),
            survivor_sel=survivor_sel,
            offspring_size=offspring_size,
            name=name,
            # Forced kwargs
            mean=mean,
            sigma=sigma,
            cov=np.eye(initializer.dimension),
            **kwargs,
        )

    def initialize(self, objfunc):
        if self.params.mean is None:
            if hasattr(objfunc, "lower_bound") and hasattr(objfunc, "upper_bound"):
                computed_mean = 0.5 * (objfunc.upper_bound + objfunc.lower_bound)
            else:
                logger.warning(
                    "Using random mean since no lower bounds could be found in the objective function. This can lead to bad convergence properties."
                )
                computed_mean = self.initializer.generate_individual()

            self.update_kwargs(mean=np.atleast_1d(computed_mean).astype(float))

        if self.params.sigma is None:
            if hasattr(objfunc, "lower_bound") and hasattr(objfunc, "upper_bound"):
                # Recommendation from: The CMA Evolution Strategy: A Tutorial (Nikolaus Hansen)
                sigma = 0.3 * (objfunc.upper_bound - objfunc.lower_bound)
            else:
                sigma = 0.5
            self.update_kwargs(sigma=np.atleast_1d(sigma).astype(float))

        # Update the operator's parameters since they were undefined in the constructor
        self.operator.update_kwargs(mean=self.params.mean, cov=self.params.cov)

        # In CMA-ES the initialization is done from random sampling of the distribution, the initializer is not used.
        mean = self.params.mean
        sigma = self.params.sigma
        cov_matrix = sigma * sigma * self.params.cov
        genotype = np.random.multivariate_normal(mean=mean, cov=cov_matrix, size=self.offspring_size)
        return Population(objfunc, genotype, encoding=self.initializer.encoding)

    def perturb(self, parents, **kwargs):

        new_mean = self.params.mean
        new_cov = self.params.cov
        self.operator.update_kwargs(mean=new_mean, cov=new_cov)

        return super().perturb(parents, **kwargs)
