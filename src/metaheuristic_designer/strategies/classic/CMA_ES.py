"""
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) implementation.

.. warning::
   The current implementation is architecturally a temporary solution.
   It will be refactored once the EDA (Distribution-based) interface is
   finalized.
"""

from __future__ import annotations
from typing import Optional
import logging
import numpy as np
import scipy as sp

from metaheuristic_designer.objective_function import ObjectiveFunc

from ...parent_selection import create_parent_selection
from ...population import Population
from ...initializer import Initializer
from ...schedulable_parameter import SchedulableParameter
from ...survivor_selection_base import SurvivorSelection
from ...operators import create_operator
from ...utils import VectorLike, check_random_state
from ..eda_strategy import EDAStrategy

logger = logging.getLogger(__name__)


class CMA_ES(EDAStrategy):
    """
    Covariance Matrix Adaptation Evolution Strategy (CMA-ES).

    This is a population-based algorithm that samples new solutions
    from a multivariate normal distribution whose mean and covariance
    are adapted each generation based on the best individuals.

    .. note::
       The architecture of this class is provisional.  It currently
       overrides :meth:`initialize` and :meth:`perturb` directly.
       Once the distribution-based (EDA) abstraction is in place,
       CMA-ES will be rewritten to use that common interface.

    Parameters
    ----------
    initializer : Initializer
        Provides population size and genotype shape, but does **not**
        generate the initial solutions.
    survivor_sel : SurvivorSelection, optional
        How survivors are selected.  Defaults to the strategy's
        default (generational).
    name : str, optional
        Display name (default ``"CMA-ES"``).
    offspring_size : int or SchedulableParameter, optional
        Number of offspring per generation.  If ``None``, the
        initializer's population size is used.
    random_state : RNGLike, optional
        Random number generator.
    mean : VectorLike, optional
        Initial mean vector.  If not given, it is computed from the
        objective's bounds (or randomly if no bounds exist).
    sigma : VectorLike, optional
        Initial step size.  If not given, a default is computed.
    **kwargs
        Forwarded to :class:`VariablePopulation`.
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
        cond_tol: float = 1e8,
        sigma_tol: float = 1e-10,
        **kwargs,
    ):
        random_state = check_random_state(random_state)

        logger.info(
            "In CMA-ES the initializer does not generate solutions, it merely indicates the population size and encoding. Don't expect different results from changing the initializer."
        )

        self.offspring_size = offspring_size if offspring_size is not None else initializer.population_size
        self.cond_tol = cond_tol
        self.sigma_tol = sigma_tol

        super().__init__(
            initializer,
            operator=create_operator(
                "mutation.full_resampling", distribution="multivariate_normal", mean=None, cov=None, allow_singular=True, random_state=random_state
            ),
            parent_sel=create_parent_selection("best", amount=initializer.population_size, random_state=random_state),
            survivor_sel=survivor_sel,
            name=name,
            random_state=random_state,
            # Forced kwargs
            sigma=sigma,
            **kwargs,
        )

        self._initialize_parameters()

    def _initialize_parameters(self):
        self._cov = np.eye(self.initializer.dimension)

        # initialize weights
        self.mu = self.initializer.population_size
        self.lambda_ = self.offspring_size

        val_range = np.arange(self.mu) + 1
        weights = np.log(self.mu + 0.5) - np.log(val_range)
        self._weights = weights / np.sum(weights)

        # Initialize internal parameters
        self._effective_pop_size = 1 / np.sum(self._weights**2)

        n = self.initializer.dimension
        norm_eff_pop = self._effective_pop_size / n
        term1 = 4 + norm_eff_pop
        term2 = n + 4 + 2 * norm_eff_pop
        self._cc = term1 / term2

        term1 = self._effective_pop_size + 2
        term2 = n + self._effective_pop_size + 5
        self._csigma = term1 / term2

        term1_a = 1 / self._effective_pop_size
        term1_b = 2 / (n + np.sqrt(2)) ** 2
        term1 = term1_a * term1_b

        term2_a = 1 - term1_a
        term3_a = 2 * self._effective_pop_size - 1
        term3_b = (n + 2) ** 2 + self._effective_pop_size
        term3 = term3_a / term3_b
        term2_b = np.minimum(term3, 1)
        term2 = term2_a * term2_b
        self._ccov = term1 + term2

        term1_a = (self._effective_pop_size - 1) / (n + 1)
        term1_b = np.sqrt(term1_a) - 1
        term1 = 2 * np.maximum(term1_b, 0)
        self._dsigma = 1 + term1 + self._csigma

        self._A = np.eye(n)

        self._xin = np.sqrt(n) * (1 - (1 / (4 * n)) + (1 / (21 * n * n)))

        # Declare internal parameters, assign dummy values
        self._path_cov = np.zeros(n)
        self._path_sigma = np.zeros(n)

        self.n = n

    def initialize(self, objfunc: ObjectiveFunc) -> Population:
        """Create the initial population by sampling from the current distribution.

        Parameters
        ----------
        objfunc : ObjectiveFunc
            The objective function, used to infer bounds if *mean* or
            *sigma* are not provided.

        Returns
        -------
        Population
            A freshly sampled population with unevaluated fitness.
        """

        if self.operator.params.mean is None:
            if hasattr(objfunc, "lower_bound") and hasattr(objfunc, "upper_bound"):
                computed_mean = 0.5 * (objfunc.upper_bound + objfunc.lower_bound)
            else:
                logger.warning(
                    "Using random mean since no lower bounds could be found in the objective function. This can lead to bad convergence properties."
                )
                computed_mean = self.initializer.generate_individual()

            if np.asarray(computed_mean).ndim == 0:
                computed_mean = np.repeat(computed_mean, self.initializer.dimension)

            self.update_kwargs(mean=np.atleast_1d(computed_mean).astype(float))

        if self.params.sigma is None:
            if hasattr(objfunc, "lower_bound") and hasattr(objfunc, "upper_bound"):
                # Recommendation from: The CMA Evolution Strategy: A Tutorial (Nikolaus Hansen)
                sigma = 0.3 * (objfunc.upper_bound - objfunc.lower_bound)
            else:
                sigma = 0.5
            self.update_kwargs(sigma=np.atleast_1d(sigma).astype(float))

        # In CMA-ES the initialization is done from random sampling of the distribution, the initializer is not used.
        mean = self.params.mean
        sigma = self.params.sigma
        cov_matrix = sigma * sigma * self._cov
        genotype = self.random_state.multivariate_normal(mean=mean, cov=cov_matrix, size=(self.offspring_size,))

        # Update the operator's parameters since they were undefined in the constructor
        self.operator.update_kwargs(mean=mean, cov=cov_matrix)

        initial_population = Population(objfunc, genotype, encoding=self.initializer.encoding)
        initial_population = initial_population.calculate_fitness()

        return initial_population

    def estimate_parameters(self, population):
        """Update the distribution parameters

        The parents (the best μ individuals from the previous generation)
        are used to update *mean*, *sigma*, *covariance*, and the
        evolution paths.

        Parameters
        ----------
        population : Population
            The selected parents (must be already evaluated).

        Returns
        -------
        Population
            Offspring population of size *offspring_size*.
        """

        pop_order = np.argsort(population.fitness)[::-1]
        pop_matrix = population.genotype_matrix[pop_order, :]

        new_mean = np.average(pop_matrix, axis=0, weights=self._weights)

        y_best = (pop_matrix - self.params.mean) / self.params.sigma
        mean_diff = (new_mean - self.params.mean) / self.params.sigma

        # Compute path values
        term1 = (1 - self._cc) * self._path_cov
        term2_a = self._cc * (2 - self._cc) * self._effective_pop_size
        term2 = np.sqrt(term2_a) * mean_diff
        self._path_cov = term1 + term2

        w = sp.linalg.solve_triangular(self._A.T, mean_diff, lower=False)

        term1 = (1 - self._csigma) * self._path_sigma
        term2_a = self._csigma * (2 - self._csigma) * self._effective_pop_size
        term2 = np.sqrt(term2_a) * w
        self._path_sigma = term1 + term2

        term1 = (1 - self._ccov) * self._cov
        term2 = (self._ccov / self._effective_pop_size) * np.outer(self._path_cov, self._path_cov)
        term3_a = self._ccov * (1 - (1 / self._effective_pop_size))
        term_b = np.zeros((self.n, self.n))
        for i in range(self.mu):
            term_b += self._weights[i] * np.outer(y_best[i], y_best[i])
        term3 = term3_a * term_b
        self._cov = term1 + term2 + term3

        self._A = np.linalg.cholesky(self._cov)

        # update sigma
        term1_a = np.linalg.norm(self._path_sigma) - self._xin
        term1_b = self._dsigma * self._xin
        new_sigma = self.params.sigma * np.exp(term1_a / term1_b)

        # Breaks to ensure numerical stability
        if np.all(new_sigma < self.sigma_tol):
            self.finish = True

        if np.linalg.cond(self._cov) > self.cond_tol:
            self.finish = True

        self.update_kwargs(sigma=new_sigma)
        self.operator.update_kwargs(mean=new_mean, cov=new_sigma * new_sigma * self._cov)

        return self.operator
