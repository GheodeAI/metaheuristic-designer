from __future__ import annotations
from ..Population import Population
from ..Initializer import Initializer
from ..ParamScheduler import ParamScheduler
from ..selectionMethods import SurvivorSelection, ParentSelection
from ..SearchStrategy import SearchStrategy
from ..Operator import Operator


class VariablePopulation(SearchStrategy):
    """
    Population-based optimization strategy where the number of individuals generated is different from the size of the population
    """

    def __init__(
        self,
        initializer: Initializer,
        operator: Operator,
        parent_sel: ParentSelection = None,
        survivor_sel: SurvivorSelection = None,
        n_offspring: int = None,
        params: ParamScheduler | dict = None,
        name: str = "Variable Population Evolution",
    ):
        self.params = params

        if n_offspring is None and initializer is not None:
            n_offspring = initializer.pop_size
        self.n_offspring = n_offspring

        self.population_shuffler = ParentSelection("Random", {"amount": self.n_offspring})

        super().__init__(
            initializer,
            operator=operator,
            parent_sel=parent_sel,
            survivor_sel=survivor_sel,
            params=params,
            name=name,
        )

    @property
    def initializer(self):
        return self._initializer

    @initializer.setter
    def initializer(self, new_initializer):
        self.n_offspring = new_initializer.pop_size
        self.population_shuffler = ParentSelection("Random", {"amount": self.n_offspring})
        self._initializer = new_initializer

    def select_parents(self, population: Population, **kwargs) -> Population:
        next_population = self.parent_sel(population)
        next_population = self.population_shuffler(next_population)
        return next_population

    def update_params(self, **kwargs):
        super().update_params(**kwargs)

        progress = kwargs["progress"]

        if isinstance(self.operator, Operator):
            self.operator.step(progress)

        if isinstance(self.survivor_sel, SurvivorSelection):
            self.survivor_sel.step(progress)

        if self.param_scheduler:
            self.param_scheduler.step(progress)
            self.params = self.param_scheduler.get_params()
