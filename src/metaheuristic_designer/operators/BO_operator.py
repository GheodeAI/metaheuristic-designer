"""
Bayesian Optimization operator based on Gaussian Process regression.
"""

from __future__ import annotations
from typing import Callable, Optional
import warnings
import numpy as np
import scipy as sp
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from ..initializer import Initializer
from ..initializers import UniformInitializer
from ..utils import RNGLike
from ..operator import Operator
from ..objective_function import ObjectiveFunc
from ..population import Population
from ..encodings import Encoding

warnings.filterwarnings(action="ignore", category=ConvergenceWarning)


def _acquisition_function(gaussian_model: GaussianProcessRegressor, _X: np.ndarray, x_in: np.ndarray, max_y: float) -> float:
    """Expected Improvement acquisition function.

    Parameters
    ----------
    gaussian_model : GaussianProcessRegressor
        The fitted GP model.
    _X : ndarray
        Training inputs (not used directly).
    x_in : ndarray
        Point where the acquisition function is evaluated.
    max_y : float
        Current best observed value.

    Returns
    -------
    float
        Negative Expected Improvement (for minimization).
    """

    mean_y, std_y = gaussian_model.predict(x_in[None, :], return_std=True)
    std_y = np.maximum(std_y, 1e-10)

    z = (mean_y - max_y) / std_y
    exp_imp = (mean_y - max_y) * sp.stats.norm.cdf(z) + std_y * sp.stats.norm.pdf(z)

    return exp_imp


class BOOperator(Operator):
    """Bayesian Optimization operator using a GP surrogate.

    Fits a Gaussian Process model to the current population, then
    maximizes the Expected Improvement acquisition function to
    propose a new candidate solution.  The new solution is merged
    back into the population.

    Parameters
    ----------
    name : str, optional
        Display name (default ``"Gaussian Regression Surrogate Model"``).
    encoding : Encoding, optional
        Encoding applied to the genotype.
    kernel : sklearn Kernel, optional
        GP kernel. Defaults to ``RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)``.
    rng : RNGLike, optional
        Random number generator.
    batch_size : int, optional
        Number of random starting points for acquisition optimization (default 100).
    max_samples : int, optional
        Maximum number of training points used (default 100).  If the
        population exceeds this, a random subset is selected.
    rbf_scale : float, optional
        Multiplicative factor applied to the RBF kernel (default 1.0).
    **kwargs
        Additional keyword arguments stored as schedulable parameters.
    """

    def __init__(
        self,
        objfunc: ObjectiveFunc,
        initializer: Initializer = None,
        name: str = "Gaussian Regression Surrogate Model",
        encoding: Optional[Encoding] = None,
        kernel: Optional[Callable] = None,
        rng: Optional[RNGLike] = None,
        batch_size: int = 100,
        max_samples: int = 100,
        rbf_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            name=name,
            encoding=encoding,
            rng=rng,
            # Forced kwargs
            batch_size=batch_size,
            max_samples=max_samples,
            **kwargs,
        )

        self.objfunc = objfunc
        if initializer is None:
            initializer = UniformInitializer(dimension=objfunc.dimension, lower_bound=objfunc.lower_bound, upper_bound=objfunc.upper_bound, rng=rng)
        self.initializer = initializer

        if kernel is None:
            kernel = rbf_scale * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        self.rbf_scale = rbf_scale

        self.gaussian_model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, copy_X_train=False)

    def evolve(self, population: Population) -> Population:
        """Fit GP, optimize acquisition, and merge the proposed point.

        Parameters
        ----------
        population : Population
            The current population.

        Returns
        -------
        Population
            The population with the new candidate appended.
        """

        # Obtain training data from the population
        population = self.objfunc.calculate_fitness(population)

        X = population.genotype_matrix
        y = population.fitness
        if population.population_size > self.params.max_samples:
            mask = self.rng.choice(population.population_size, size=self.params.max_samples, replace=False)
            X = X[mask]
            y = y[mask]

        # Fit the surrogate model
        self.gaussian_model.fit(X, y)

        # Initialize optimization data structures
        max_y = np.max(self.gaussian_model.predict(X))
        min_ei = float("inf")
        new_best_point = X[0]

        if isinstance(self.objfunc, ObjectiveFunc):
            bounds = np.asarray((self.objfunc.lower_bound, self.objfunc.upper_bound)).T
            if bounds.ndim == 1:
                bounds = bounds[None, :]
        else:
            bounds = None

        # Optimize the acquisition function with a batch of initial points chosen at random
        samples = self.initializer.generate_population(self.params.batch_size).genotype_matrix
        for x0 in samples:
            result = sp.optimize.minimize(
                fun=lambda x_in: -_acquisition_function(self.gaussian_model, X, x_in, max_y), x0=x0, method="L-BFGS-B", bounds=bounds
            )
            if result.fun < min_ei:
                min_ei = result.fun
                new_best_point = result.x

        # Create new population from the optimization result and merge it with the previous one
        new_sample_population = Population(genotype_matrix=new_best_point[None, :], encoding=population.encoding)
        new_population = Population.join_populations(population, new_sample_population)
        new_population = self.objfunc.repair_solutions(new_population)

        return new_population
