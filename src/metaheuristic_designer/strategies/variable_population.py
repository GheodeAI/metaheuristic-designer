from __future__ import annotations
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


class VariablePopulation(SearchStrategy):
    """
    Population-based optimization strategy where the number of individuals generated is different from the size of the population
    """

    def __init__(
        self,
        initializer: Initializer,
        operator: Operator,
        parent_sel: Optional[ParentSelection] = None,
        survivor_sel: Optional[SurvivorSelection] = None,
        offspring_size: Optional[int | SchedulableParameter] = None,
        name: str = "Variable Population Evolution",
        random_state: Optional[RNGLike] = None,
        **kwargs,
    ):
        self.using_custom_offspring_size = offspring_size is not None

        if not self.using_custom_offspring_size:
            offspring_size = initializer.pop_size

        # We need to set up the random state beforehand to handle the initializer correctly
        self.random_state = check_random_state(random_state)
        self.population_shuffler = create_parent_selection("Random", amount=offspring_size, random_state=self.random_state)

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
    def initializer(self):
        return self._initializer

    @initializer.setter
    def initializer(self, new_initializer):
        if not self.using_custom_offspring_size:
            self.update_kwargs(offspring_size=new_initializer.pop_size)
            self.population_shuffler = create_parent_selection("Random", amount=self.params.offspring_size, random_state=self.random_state)
        self._initializer = new_initializer

    def select_parents(self, population: Population) -> Population:
        next_population = super().select_parents(population)
        next_population = self.population_shuffler(next_population)
        return next_population
