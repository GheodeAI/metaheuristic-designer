"""
Single solution abstract strategy
"""

from __future__ import annotations
from copy import copy
from typing import Optional

from ..population import Population
from ..objective_function import ObjectiveFunc
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
    rng : RNGLike, optional
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
        rng: Optional[RNGLike] = None,
        **kwargs,
    ):
        super().__init__(initializer, operator=operator, survivor_sel=survivor_sel, name=name, rng=rng, **kwargs)

    def step(self, prev_population: Population, objfunc: ObjectiveFunc) -> Population:
        population = copy(prev_population)
        population = self.operator.evolve(population)
        population = objfunc.repair_solutions(population)
        population = objfunc.calculate_fitness(population)
        population = self.survivor_sel.select(population=prev_population, offspring=population)
        return population
