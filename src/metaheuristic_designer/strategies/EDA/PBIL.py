from __future__ import annotations
import numpy as np
from ...operators import create_mutation_operator
from ...parent_selection import ParentSelection
from ...survivor_selection import SurvivorSelection
from ...initializer import Initializer
from ..variable_population import VariablePopulation
from ...utils import check_random_state


class BernoulliPBIL(VariablePopulation):
    """
    Estimation of distribution algorithm for binary vectors.
    https://doi.org/10.1016/j.swevo.2011.08.003
    """

    def __init__(
        self,
        initializer: Initializer,
        parent_sel: ParentSelection = None,
        survivor_sel: SurvivorSelection = None,
        params: dict = None,
        name: str = "BernoulliPBIL",
        random_state=None,
    ):
        if params is None:
            params = {}

        self.random_state = check_random_state(random_state)

        self.p = params.get("p", None)

        evolve_op = create_mutation_operator("RandSample", distrib="bernoulli", p=self.p)
        offspring_size = params.get("offspringSize", initializer.pop_size)

        self.lr = params.get("lr")
        self.noise = params.get("noise", 0)

        super().__init__(
            initializer,
            evolve_op,
            parent_sel=parent_sel,
            survivor_sel=survivor_sel,
            offspring_size=offspring_size,
            params=params,
            name=name,
        )

    def _batch_fit(self, population):
        population_matrix = population.genotype_matrix
        p_hat = population_matrix.mean(axis=0)

        return p_hat

    def perturb(self, parents, **kwargs):
        new_p = self._batch_fit(parents)
        if self.p is not None:
            self.p = (1 - self.lr) * self.p + self.lr * new_p
            self.p += self.random_state.normal(0, self.noise, size=self.p.shape)
            self.p = np.clip(self.p, 0, 1)
        else:
            self.p = new_p

        self.operator = create_mutation_operator("RandSample", distrib="bernoulli", p=self.p)

        return super().perturb(parents, **kwargs)


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
        params: dict = None,
        name: str = "BernoulliPBIL",
        random_state=None,
    ):
        if params is None:
            params = {}

        self.random_state = check_random_state(random_state)

        self.p = params.get("p", None)

        if "n" not in params:
            raise Exception("A parameter 'n' must be specified which indicates the maximum value of the Binomial distribution.")

        self.n = params["n"]

        evolve_op = create_mutation_operator("RandSample", distrib="binomial", p=self.p, n=self.n)
        self.prob_vec_mutate = evolve_op

        offspring_size = params.get("offspringSize", initializer.pop_size)

        self.lr = params.get("lr")
        self.noise = params.get("noise", 0)

        super().__init__(
            initializer,
            evolve_op,
            parent_sel=parent_sel,
            survivor_sel=survivor_sel,
            offspring_size=offspring_size,
            params=params,
            name=name,
        )

    def _batch_fit(self, population):
        population_matrix = population.genotype_matrix
        p_hat = population_matrix.sum(axis=0) / (self.n * population_matrix.shape[0])

        return p_hat

    def perturb(self, parents, **kwargs):
        new_p = self._batch_fit(parents)
        if self.p is not None:
            self.p = (1 - self.lr) * self.p + self.lr * new_p
            self.p += self.random_state.normal(0, self.noise, size=self.p.shape)
            self.p = np.clip(self.p, 0, 1)
        else:
            self.p = new_p

        self.operator = create_mutation_operator("RandSample", distrib="binomial", p=self.p, n=self.n)

        return super().perturb(parents, **kwargs)


class GaussianPBIL(VariablePopulation):
    """
    Estimation of distribution algorithm for binary vectors.
    https://doi.org/10.1016/j.swevo.2011.08.003
    """

    def __init__(
        self,
        initializer: Initializer,
        parent_sel: ParentSelection = None,
        survivor_sel: SurvivorSelection = None,
        params: dict = None,
        name: str = "GaussianPBIL",
        random_state=None,
    ):
        if params is None:
            params = {}

        self.random_state = check_random_state(random_state)

        self.loc = params.get("loc", None)
        self.scale = params.get("scale", 1)

        evolve_op = create_mutation_operator("RandSample", distrib="gaussian", loc=self.loc, scale=self.scale)
        offspring_size = params.get("offspringSize", initializer.pop_size)

        self.lr = params.get("lr")
        self.noise = params.get("noise", 0)

        super().__init__(
            initializer,
            evolve_op,
            parent_sel=parent_sel,
            survivor_sel=survivor_sel,
            offspring_size=offspring_size,
            params=params,
            name=name,
        )

    def _batch_fit(self, population):
        population_matrix = population.genotype_matrix
        loc_hat = population_matrix.mean(axis=0)

        return loc_hat

    def perturb(self, parents, **kwargs):
        new_loc = self._batch_fit(parents)
        if self.loc is not None:
            self.loc = (1 - self.lr) * self.loc + self.lr * new_loc
            self.loc += self.random_state.normal(0, self.noise, size=self.loc.shape)
        else:
            self.loc = new_loc

        self.operator = create_mutation_operator("RandSample", distrib="gaussian", loc=self.loc, scale=self.scale)

        return super().perturb(parents, **kwargs)
