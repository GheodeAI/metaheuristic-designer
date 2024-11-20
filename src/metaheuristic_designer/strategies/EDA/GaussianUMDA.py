from __future__ import annotations
import numpy as np
import scipy as sp
from ...operators import OperatorVector
from ...selectionMethods import ParentSelection, SurvivorSelection
from ...Initializer import Initializer
from ...ParamScheduler import ParamScheduler
from ..VariablePopulation import VariablePopulation
from ...utils import RAND_GEN


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
        params: ParamScheduler | dict = {},
        name: str = "GaussianUMDA",
    ):
        self.loc = params.get("loc", 0)
        self.scale = params.get("scale", 1)

        evolve_op = OperatorVector("RandSample", {"distrib": "Gaussian", "loc": self.loc, "scale": self.scale})
        offspring_size = params.get("offspringSize", initializer.pop_size)

        self.noise = params.get("noise", 0)

        super().__init__(
            initializer,
            evolve_op,
            parent_sel=parent_sel,
            survivor_sel=survivor_sel,
            n_offspring=offspring_size,
            params=params,
            name=name,
        )

    def _batch_fit(self, population):
        population_matrix = population.genotype_set
        loc_hat = population_matrix.mean(axis=0)

        return loc_hat

    def perturb(self, parents, **kwargs):
        self.loc = self._batch_fit(parents)
        self.loc += RAND_GEN.normal(0, self.noise, size=self.loc.shape)

        self.operator = OperatorVector("RandSample", {"distrib": "Gaussian", "loc": self.loc, "scale": self.scale})

        return super().perturb(parents, **kwargs)
