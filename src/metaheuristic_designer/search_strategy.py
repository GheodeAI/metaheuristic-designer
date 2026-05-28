"""
Base class for the Search strategy module.

This module implements the procedure applied in each iteration of the algorithm.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import logging
from typing import Optional, Callable

from .parent_selection_base import ParentSelection, NullParentSelection
from .survivor_selection_base import SurvivorSelection, NullSurvivorSelection
from .population import Population
from .initializer import Initializer, InitializerFromLambda
from .objective_function import ObjectiveFunc
from .operator import Operator, NullOperator
from .parametrizable_mixin import ParametrizableMixin
from .utils import check_random_state, RNGLike

logger = logging.getLogger(__name__)


class SearchStrategy(ParametrizableMixin, ABC):
    """Orchestrates one iteration of an optimization loop.

    A search strategy holds together an :class:`Initializer`, an
    :class:`Operator`, a :class:`ParentSelection`, and a
    :class:`SurvivorSelection`.  Together they define how the
    population is created, perturbed, and pruned each generation.
    Subclasses can override any step to implement algorithm-specific
    logic.

    Parameters
    ----------
    initializer : Initializer
        Creates the starting population.
    operator : Operator, optional
        The perturbation operator (mutation, crossover, …).
        Defaults to :class:`NullOperator`.
    parent_sel : ParentSelection, optional
        Selects which individuals are used to generate offspring.
        Defaults to :class:`NullParentSelection`.
    survivor_sel : SurvivorSelection, optional
        Selects which individuals survive to the next generation.
        Defaults to :class:`NullSurvivorSelection`.
    name : str, optional
        Display name used in reports.
    random_state : RNGLike, optional
        Random number generator.
    **kwargs
        Additional keyword arguments stored as schedulable
        parameters.
    """

    def __init__(
        self,
        initializer: Initializer,
        operator: Optional[Operator] = None,
        parent_sel: Optional[ParentSelection] = None,
        survivor_sel: Optional[SurvivorSelection] = None,
        name: str = "some strategy",
        random_state: Optional[RNGLike] = None,
        **kwargs,
    ):
        super().__init__()

        self.random_state = check_random_state(random_state)

        self.name = name
        self.initializer = initializer

        if operator is None:
            operator = NullOperator()
        self.operator = operator

        if parent_sel is None:
            parent_sel = NullParentSelection()
        self.parent_sel = parent_sel

        if survivor_sel is None:
            survivor_sel = NullSurvivorSelection()
        self.survivor_sel = survivor_sel

        self.best = None
        self.finish = False
        self.store_kwargs(**kwargs)

    @property
    def population_size(self) -> int:
        """
        Gets the amount of individuals in the population.
        """

        return self.initializer.population_size

    def gather_parameters(self):
        """Collect the current parameters from all sub-components.

        Returns
        -------
        dict
            A flat dictionary with dotted keys like
            ``"operator.F"``, ``"parent_sel.amount"``, etc.
        """

        param_dict = {f"{self.parent_sel.name}.{k}": v for k, v in self.parent_sel.gather_params().items()}
        param_dict.update({f"{self.operator.name}.{k}": v for k, v in self.operator.gather_params().items()})
        param_dict.update({f"{self.survivor_sel.name}.{k}": v for k, v in self.survivor_sel.gather_params().items()})
        return param_dict

    def reset(self, objfunc: ObjectiveFunc):
        objfunc.reset()

    def initialize(self, objfunc: ObjectiveFunc) -> Population:
        """
        Initializes the optimization search strategy.

        Parameters
        ----------
        objfunc: ObjectiveFunc
            Objective function to be optimized.

        Returns
        -------
        population: Population
            The initial population to be used in the algorithm.
        """

        initial_population = self.initializer.generate_population(objfunc)
        initial_population = initial_population.calculate_fitness()
        return initial_population

    def update(self, progress: float):
        """Advances the state of the search by one iteration.

        Parameters
        ----------
        progress : float
            Current progress of the algorithm (0-1).
        """

        super().update(progress)
        self.operator.update(progress)
        self.parent_sel.update(progress)
        self.survivor_sel.update(progress)

    @abstractmethod
    def step(self, prev_population: Population) -> Population:
        """Performs a single iteration of the algorithm on a given population.

        Parameters
        ----------
        population : Population
            Population of solutions in which to perform the operators.

        Returns
        -------
        Population
            Next population after performing all the steps in the iteration.
        """

    def get_state(self) -> dict:
        """
        Gets the current state of the search strategy as a dictionary.

        Parameters
        ----------
        show_population: bool, optional
            Save the state of the current population.

        Returns
        -------
        state: dict
            The complete state of the search strategy.
        """

        data = {
            "class_name": type(self).__name__,
            "name": self.name,
            "random_generator": type(self.random_state).__name__,
            "random_state": self.random_state.bit_generator.state,
            "initializer": self.initializer.get_state(),
            "parent_sel": self.parent_sel.get_state(),
            "operators": self.operator.get_state(),
            "survivor_sel": self.survivor_sel.get_state(),
            **self.get_params(),
        }

        return data

    def extra_step_info(self):
        """Hook called after each generation (intended for subclasses)."""

    def extra_report(self):
        """Hook called at the end of the optimization (intended for subclasses)."""


class SearchStrategyFromLambda(SearchStrategy):
    """Strategy whose components can be plain functions.

    Accepts each component as either a properly constructed object
    or a callable; if a callable is provided it is automatically
    wrapped with the appropriate ``*FromLambda`` class.  This is
    the simplest way to build a custom strategy in one go.

    Parameters
    ----------
    initializer : callable or Initializer
        Function ``(random_state) -> genotype``, or an initializer
        instance.
    iterate_fn: callable
        Function that advances the state of the algorithm by one full iteration.
    name : str, optional
        Display name (default ``"Strategy from lambda"``).
    random_state : RNGLike, optional
        Random number generator.
    **kwargs
        Forwarded to :class:`SearchStrategy`.
    """

    def __init__(
        self,
        initializer: Callable | Initializer,
        iterate_fn: Callable,
        name: str = "Custom strategy",
        random_state: Optional[RNGLike] = None,
        **kwargs,
    ):
        if not isinstance(initializer, Initializer):
            initializer = InitializerFromLambda(initializer)

        self.iterate_fn = iterate_fn

        super().__init__(
            initializer=initializer,
            name=name,
            random_state=random_state,
            **kwargs,
        )

    def step(self, population):
        return self.iterate_fn(population)
