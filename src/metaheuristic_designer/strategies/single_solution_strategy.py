"""
Single solution abstract strategy
"""

from __future__ import annotations
from copy import copy
from typing import Optional

from metaheuristic_designer.population import Population
from ..initializer import Initializer
from ..survivor_selection_base import SurvivorSelection
from ..search_strategy import SearchStrategy
from ..operator import Operator
from ..utils import RNGLike


class SingleSolutionStrategy(SearchStrategy):
    """

    No parent selection method exists, we only have one solution at each given time

    Parameters
    ----------
    initializer : Initializer
        Population initializer (typically creates a single individual).
    operator : Operator, optional
        Perturbation operator.  Defaults to :class:`NullOperator`.
    survivor_sel : SurvivorSelection, optional
        Survivor selection method; defaults to ``"hill_climb"``.
    name : str, optional
        Display name (default ``"HillClimb"``).
    random_state : RNGLike, optional
        Random number generator.
    **kwargs
        Forwarded to :class:`SearchStrategy`.
    """

    def __init__(
        self,
        initializer: Initializer,
        operator: Optional[Operator] = None,
        survivor_sel: Optional[SurvivorSelection] = None,
        name: str = "HillClimb",
        random_state: Optional[RNGLike] = None,
        **kwargs,
    ):
        super().__init__(initializer, operator=operator, survivor_sel=survivor_sel, name=name, random_state=random_state, **kwargs)

    def step(self, prev_population: Population) -> Population:
        self.population = copy(prev_population)
        self.population = self.operator.evolve(self.population, self.initializer)
        self.population = self.population.repair_solutions()
        self.population = self.population.calculate_fitness()
        self.population = self.survivor_sel.select(population=prev_population, offspring=self.population)
        return self.population
