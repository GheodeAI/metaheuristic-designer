from __future__ import annotations
import numpy as np
import scipy as sp
from ...Individual import Individual
from ...operators import OperatorVector
from ...selectionMethods import ParentSelection, SurvivorSelection
from ...Initializer import Initializer
from ...ParamScheduler import ParamScheduler
from ..VariablePopulation import VariablePopulation
from ...utils import RAND_GEN


class BinomialPBIL(VariablePopulation):
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
        name: str = "BernoulliPBIL",
    ):
        self.p = params.get("p", None)
        if "n" not in params:
            raise Exception("A parameter 'n' must be specified which indicates the maximum value.")

        self.n = params["n"]

        evolve_op = OperatorVector("RandSample", {"distrib": "Bernoulli", "p": self.p, "n": self.n})
        self.prob_vec_mutate = evolve_op

        offspring_size = params.get("offspringSize", initializer.pop_size)

        self.lr = params.get("lr")
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

    def _batch_fit(self, parent_list):
        population_matrix = np.asarray([i.genotype for i in parent_list])
        p_hat = population_matrix.sum(axis=0) / (self.n * population_matrix.shape[0])

        return p_hat

    def perturb(self, parent_list, objfunc, **kwargs):
        new_p = self._batch_fit(parent_list)
        if self.p is not None:
            self.p = (1 - self.lr) * self.p + self.lr * new_p
            self.p += RAND_GEN.normal(0, self.noise, size=self.p.shape)
            self.p = np.clip(self.p, 0, 1)
        else:
            self.p = new_p

        self.operator = OperatorVector("RandSample", {"distrib": "Bernoulli", "p": self.p, "n": self.n})

        return super().perturb(parent_list, objfunc, **kwargs)
