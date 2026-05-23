"""
Strategy where the population size remains constant, no explicit parent selection.
"""

from __future__ import annotations
from copy import copy
from typing import Optional

from metaheuristic_designer.population import Population
from ..initializer import Initializer
from ..parent_selection_base import ParentSelection
from ..survivor_selection_base import SurvivorSelection
from ..search_strategy import SearchStrategy
from ..operator import Operator
from ..utils import RNGLike


class StaticPopulationStrategy(SearchStrategy):
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
    random_state : RNGLike, optional
        Random number generator.
    **kwargs
        Forwarded to :class:`SearchStrategy`.
    """

    def __init__(
        self,
        initializer: Initializer,
        operator: Operator = None,
        parent_sel: Optional[ParentSelection] = None,
        survivor_sel: Optional[SurvivorSelection] = None,
        name: str = "Static Population Evolution",
        random_state: Optional[RNGLike] = None,
        **kwargs,
    ):
        super().__init__(
            initializer, operator=operator, parent_sel=parent_sel, survivor_sel=survivor_sel, name=name, random_state=random_state, **kwargs
        )

    def step(self, prev_population: Population) -> Population:
        population = self.parent_sel.select(prev_population)  # implicit copy
        population = self.operator.evolve(population, self.initializer)
        population = population.repair_solutions()
        population = population.calculate_fitness()
        population = self.survivor_sel.select(population=prev_population, offspring=population)
        return population
