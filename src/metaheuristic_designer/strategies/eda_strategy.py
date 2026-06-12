"""
Strategy that generates solutions from a model.
"""

from __future__ import annotations
from abc import abstractmethod
from typing import Optional

from ..population import Population
from ..objective_function import ObjectiveFunc
from ..initializer import Initializer
from ..parent_selection_base import ParentSelection
from ..survivor_selection_base import SurvivorSelection
from ..search_strategy import SearchStrategy
from ..operator import Operator
from ..utils import RNGLike


class EDAStrategy(SearchStrategy):
    """
    Population-based strategy with constant size and no parent selection.

    The entire population is perturbed each generation.  By default,
    parent selection is the identity (all individuals are used) and
    survivor selection is generational (offspring replace parents).

    Parameters
    ----------
    initializer : Initializer
        Population initializer.
    operator : Operator
        Perturbation operator.
    parent_sel : ParentSelection, optional
        Parent selection; defaults to identity (keep all).
    survivor_sel : SurvivorSelection, optional
        Survivor selection; defaults to generational replacement.
    name : str, optional
        Display name (default ``"Static Population Evolution"``).
    rng : RNGLike, optional
        Random number generator.
    **kwargs
        Forwarded to :class:`SearchStrategy`.
    """

    def __init__(
        self,
        initializer: Initializer,
        operator: Operator,
        parent_sel: Optional[ParentSelection] = None,
        survivor_sel: Optional[SurvivorSelection] = None,
        name: str = "Static Population Evolution",
        rng: Optional[RNGLike] = None,
        **kwargs,
    ):
        self.sampler = initializer
        super().__init__(
            initializer=initializer,
            operator=operator,
            parent_sel=parent_sel,
            survivor_sel=survivor_sel,
            name=name,
            rng=rng,
            **kwargs,
        )

    @abstractmethod
    def estimate_parameters(self, population: Population) -> Operator:
        """Utilizes the samples present in the input population to
        estimate the parameters used by the operator.

        Parameters
        ----------
        population : Population
            Data to use for estimating parameters.

        Returns
        -------
        Operator
            Newly configured operator.
        """

    def step(self, prev_population: Population, objfunc: ObjectiveFunc) -> Population:
        population = self.parent_sel.select(prev_population)
        self.operator = self.estimate_parameters(population)
        population = self.operator.evolve(population)
        population = objfunc.repair_population(population)
        population = objfunc.calculate_fitness(population)
        population = self.survivor_sel.select(population=prev_population, offspring=population)
        return population
