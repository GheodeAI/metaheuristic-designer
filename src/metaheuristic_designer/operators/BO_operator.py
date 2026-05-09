from __future__ import annotations
import warnings
import numpy as np
import scipy as sp
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from ..operator import Operator
from ..objective_function import VectorObjectiveFunc
from ..population import Population

warnings.filterwarnings(action="ignore", category=ConvergenceWarning)


def _acquisition_function(gaussian_model, _X, x_in, max_y):
    mean_y, std_y = gaussian_model.predict(x_in[None, :], return_std=True)
    std_y = np.maximum(std_y, 1e-10)

    z = (mean_y - max_y) / std_y
    exp_imp = (mean_y - max_y) * sp.stats.norm.cdf(z) + std_y * sp.stats.norm.pdf(z)

    return exp_imp


class BOOperator(Operator):
    """
    Operator used specifically in the Bayesian Optimization algorithm.

    Implements a surrogate model based on a Gaussian Process Regressor. An aquisition function calculated
    from the regression model is then optimized to estimate the next best solution for the problem.
    """

    def __init__(self, name="Gaussian Regression Surrogate Model", encoding=None, kernel=None, random_state=None, batch_size = 100, max_samples=100, rbf_scale=1.0, **kwargs):
        super().__init__(
            name=name,
            encoding=encoding,
            random_state=random_state,

            # Forced kwargs
            batch_size = batch_size,
            max_samples = max_samples,
            **kwargs
        )

        if kernel is None:
            kernel = rbf_scale * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        self.rbf_scale = rbf_scale

        self.gaussian_model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, copy_X_train=False)

    def evolve(self, population, initializer=None):
        # Obtain training data from the population
        population = population.calculate_fitness()

        X = population.genotype_matrix
        y = population.fitness
        if population.pop_size > self.params.max_samples:
            mask = self.random_state.choice(population.pop_size, size=self.params.max_samples, replace=False)
            X = X[mask]
            y = y[mask]

        # Fit the surrogate model
        self.gaussian_model.fit(X, y)

        # Initialize optimization data structures
        objfunc = population.objfunc
        max_y = np.max(self.gaussian_model.predict(X))
        min_ei = float("inf")
        new_best_point = X[0]

        if isinstance(objfunc, VectorObjectiveFunc):
            bounds = np.asarray((objfunc.lower_bound, objfunc.upper_bound)).T
            if bounds.ndim == 1:
                bounds = bounds[None, :]
        else:
            bounds = None

        # Optimize the acquisition function with a batch of initial points chosen at random
        samples = initializer.generate_population(objfunc, self.params.batch_size).genotype_matrix
        for x0 in samples:
            result = sp.optimize.minimize(
                fun=lambda x_in: -_acquisition_function(self.gaussian_model, X, x_in, max_y), x0=x0, method="L-BFGS-B", bounds=bounds
            )
            if result.fun < min_ei:
                min_ei = result.fun
                new_best_point = result.x

        # Create new population from the optimization result and merge it with the previous one
        new_sample_population = Population(objfunc, genotype_matrix=new_best_point[None, :], encoding=population.encoding)
        new_population = Population.join_populations(population, new_sample_population)
        new_population = new_population.repair_solutions()

        return new_population
