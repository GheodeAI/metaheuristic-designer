"""
Strategy where offspring size differs from population size (μ+λ / μ,λ style).
"""

from __future__ import annotations
import logging
from typing import Optional
from ..population import Population
from ..initializer import Initializer
from ..parent_selection_base import ParentSelection
from ..survivor_selection_base import SurvivorSelection
from ..parent_selection import create_parent_selection
from ..search_strategy import SearchStrategy
from ..operator import Operator
from ..utils import check_random_state, RNGLike
from ..schedulable_parameter import SchedulableParameter

logger = logging.getLogger(__name__)


class ShuffledPopulationStrategy(SearchStrategy):
    """
    Population-based strategy with separate parent and offspring sizes.

    This is the base for (μ+λ) and (μ,λ) Evolution Strategies, GAs, and
    similar algorithms.  The number of parents selected and the number of
    offspring generated can be configured independently.

    Parameters
    ----------
    initializer : Initializer
        Population initializer.
    operator : Operator
        Perturbation operator.
    parent_sel : ParentSelection, optional
        Parent selection method.
    survivor_sel : SurvivorSelection, optional
        Survivor selection method.
    offspring_size : int or SchedulableParameter, optional
        Number of offspring to generate.  Defaults to the
        initializer's population size.
    shuffle_with_replacement : bool, optional
        If ``True``, shuffle the parent pool with replacement;
        otherwise without replacement (default ``False``).
    name : str, optional
        Display name.
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
        offspring_size: Optional[int | SchedulableParameter] = None,
        shuffle_with_replacement: bool = False,
        name: str = "Variable Population Evolution",
        random_state: Optional[RNGLike] = None,
        **kwargs,
    ):
        # We need to set up the random state beforehand to handle the initializer correctly
        random_state = check_random_state(random_state)

        self.using_custom_offspring_size = offspring_size is not None

        if offspring_size is None:
            offspring_size = initializer.population_size
        self.offspring_size = offspring_size

        self.shuffle_with_replacement = shuffle_with_replacement

        super().__init__(
            initializer,
            operator=operator,
            parent_sel=parent_sel,
            survivor_sel=survivor_sel,
            name=name,
            random_state=random_state,
            # Forced kwargs
            offspring_size=offspring_size,
            **kwargs,
        )

    @property
    def initializer(self) -> Initializer:
        return self._initializer

    @initializer.setter
    def initializer(self, new_initializer: Initializer):
        """Update the offspring size and shuffler when the initializer changes.

        Parameters
        ----------
        new_initializer : Initializer
            The new initializer.
        """

        if not self.using_custom_offspring_size:
            self.update_kwargs(offspring_size=new_initializer.population_size)

        if hasattr(self.params, "offspring_size"):
            offspring_size = self.params.offspring_size
        else:
            offspring_size = self.offspring_size

        if self.shuffle_with_replacement:
            self.population_shuffler = create_parent_selection("random_with_replacement", amount=offspring_size, random_state=self.random_state)
        else:
            self.population_shuffler = create_parent_selection("random_without_replacement", amount=offspring_size, random_state=self.random_state)

        self._initializer = new_initializer

    def step(self, prev_population: Population) -> Population:
        population = self.parent_sel.select(prev_population)  # implicit copy
        population = self.population_shuffler(population)
        population = self.operator.evolve(population, self.initializer)
        population = population.repair_solutions()
        population = population.calculate_fitness()
        population = self.survivor_sel.select(population=prev_population, offspring=population)
        return population
