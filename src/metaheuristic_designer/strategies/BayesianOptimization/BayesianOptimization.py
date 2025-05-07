from __future__ import annotations
from copy import copy
import numpy as np
import scipy as sp
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from ...Initializer import Initializer
from ...ParamScheduler import ParamScheduler
from ...selectionMethods import SurvivorSelection, SurvivorSelectionNull
from ..HillClimb import HillClimb
import warnings
warnings.filterwarnings(action="ignore", category=ConvergenceWarning)

class BayesianOptimization(HillClimb):
    """
    Simulated annealing
    """

    def __init__(
        self,
        initializer: Initializer,
        params: ParamScheduler | dict = None,
        name: str = "SA",
    ):
        # kernel = RBF() + WhiteKernel()
        # self.gaussian_model = GaussianProcessRegressor(kernel=kernel)
        self.gaussian_model = GaussianProcessRegressor()
        self.batch_size = 100
        self.scale = 100

        super().__init__(
            initializer,
            survivor_sel=SurvivorSelectionNull(),
            params=params,
            name=name
        )
    
    @staticmethod
    def _aquisition_function(gaussian_model, X, x_in):
        
        mean_y, std_y = gaussian_model.predict(x_in[None, :], return_std=True)
        std_y = std_y[None, :]
        if std_y == 0:
            return 0

        mean_y_init = gaussian_model.predict(X)
        max_y = np.max(mean_y_init)

        z = (mean_y - max_y) / std_y
        exp_imp = (mean_y - max_y) * sp.stats.norm.cdf(z) + std_y * sp.stats.norm.pdf(z)

        return exp_imp


    def perturb(self, parents: Population, **kwargs) -> Population:
        X = parents.genotype_set
        y = parents.calculate_fitness()

        self.gaussian_model.fit(X, y)
        yhat = self.gaussian_model.predict(X)

        min_ei = float("-inf")
        samples = self.initializer.generate_population(None, self.batch_size).genotype_set
        for x0 in samples:
            result = sp.optimize.minimize(
                fun=lambda x0: self._aquisition_function(self.gaussian_model, X, x0),
                x0=x0,
                method="L-BFGS-B"
            )
            if result.fun > min_ei:
                min_ei = result.fun
                new_best_point = result.x
                
        new_X = np.vstack([X, new_best_point[None, :]])
        new_parents = copy(parents)
        new_parents.update_genotype_set(new_X)

        return new_parents