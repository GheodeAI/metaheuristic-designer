"""
Base class for the Search strategy module.

This module implements the procedure applied in each iteration of the algorithm.
"""

from __future__ import annotations
import logging
from typing import Tuple, Any, Optional, Callable

from .parent_selection_base import ParentSelection, NullParentSelection, ParentSelectionFromLambda
from .survivor_selection_base import SurvivorSelection, NullSurvivorSelection, SurvivorSelectionFromLambda
from .population import Population
from .initializer import Initializer
from .objective_function import ObjectiveFunc
from .operator import Operator, NullOperator, OperatorFromLambda
from .parametrizable_mixin import ParametrizableMixin
from .utils import check_random_state, RNGLike
from .initializer import InitializerFromLambda
from .encoding import Encoding, EncodingFromLambda

logger = logging.getLogger(__name__)


class SearchStrategy(ParametrizableMixin):
    """Orchestrates one iteration of an optimisation loop.

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

        self.population = None
        self.best = None
        self.finish = False
        self.random_state = check_random_state(random_state)
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
        if self.population is not None:
            param_dict.update({f"{self.population.encoding.name}.{k}": v for k, v in self.population.encoding.gather_params().items()})
        return param_dict

    def best_solution(self) -> Tuple[Any, float]:
        """
        Returns the best solution found by the search strategy and its fitness.

        Returns
        -------
        best_solution : Tuple[Any, float]
            A pair of the best individual with its fitness.
        """

        return self.population.best_solution()

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
            The initial population to be used in the algoritm.
        """

        return self.initializer.generate_population(objfunc)

    def evaluate_population(self, population: Population, parallel: bool = False, threads: int = 8) -> Population:
        """
        Calculates the fitness of the individuals on the population.

        Parameters
        ----------
        population: Population
        parallel: bool, optional
            Whether to evaluate the individuals in the population in parallel.
        threads: int, optional
            Number of processes to use at once if calculating the fitness in parallel.

        Returns
        -------
        population: Population
            The population with the fitness values recorded.
        """

        return population.calculate_fitness(parallel=parallel, threads=threads)

    def select_parents(self, population: Population, amount: Optional[int] = None) -> Population:
        """
        Selects the individuals that will be perturbed in this generation to generate the offspring.

        Parameters
        ----------
        population: Population
            The current population of the search strategy.

        Returns
        -------
        parents: Population
            A pair of the list of individuals considered as parents and their position in the original population.
        """

        logger.debug("Selected parents...")
        return self.parent_sel(population, amount=amount)

    def perturb(self, parents: Population, **kwargs) -> Population:
        """
        Applies operators to the population to get the next generation of individuals.

        Parameters
        ----------
        parents: Population
            The current parents that will be used in the search strategy.

        Returns
        -------
        offspring: Population
            The list of individuals modified by the operators of the search strategy.
        """

        offspring = self.operator.evolve(parents, self.initializer)
        offspring = self.repair_population(offspring)

        logger.debug("Applied perturbation operators...")
        return offspring

    def repair_population(self, population: Population) -> Population:
        """
        Repairs the individuals in the population to make them fulfill the problem's restrictions.

        Parameters
        ----------
        population: Population
            The population to be repaired

        Returns
        -------
        repaired_population: Population
            The population of repaired individuals
        """

        logger.debug("Applied hard constraints...")
        return population.repair_solutions()

    def select_individuals(self, population: Population, offspring: Population, **kwargs) -> Population:
        """
        Selects the individuals that will pass to the next generation.

        Parameters
        ----------
        population: Population
            The current population of the search strategy.
        offspring: Population
            The list of individuals modified by the operators of the search strategy.

        Returns
        -------
        offspring: Population
            The list of individuals selected for the next generation.
        """

        logger.debug("Selected survivors...")
        return self.survivor_sel(population, offspring)

    def step(self, progress: float):
        """Update internal parameters and forward progress to sub-components.

        Parameters
        ----------
        progress : float
            Current progress of the algorithm (0-1).
        """

        super().step(progress)

        self.operator.step(progress)
        self.parent_sel.step(progress)
        self.survivor_sel.step(progress)

    def get_state(self, store_population: bool = False) -> dict:
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
            "population": self.population.get_state() if store_population else None,
            **self.get_params(),
        }

        return data

    def extra_step_info(self):
        """Hook called after each generation (intended for subclasses)."""

    def extra_report(self):
        """Hook called at the end of the optimisation (intended for subclasses)."""


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
    operator : callable or Operator, optional
        Function ``(population, initializer, random_state, **kwargs) -> Population``,
        or an operator instance.
    parent_sel : callable or ParentSelection, optional
        Function ``(population, amount, random_state, **kwargs) -> indices``,
        or a selection instance.
    survivor_sel : callable or SurvivorSelection, optional
        Function ``(parent_fitness, offspring_fitness, random_state, **kwargs) -> indices``,
        or a selection instance.
    name : str, optional
        Display name (default ``"Strategy from lambda"``).
    encoding : Encoding, optional
        Encoding that wraps encode/decode; overridden by
        *encode_fn*/*decode_fn* if both are given.
    encode_fn / decode_fn : callable, optional
        Standalone encode/decode functions.
    parent_selection_amount : int, optional
        Amount of parents to select (used only when wrapping a
        callable *parent_sel*).
    pop_size : int, optional
        Population size (used only when wrapping a callable
        *initializer*).  Default 100.
    random_state : RNGLike, optional
        Random number generator.
    **kwargs
        Forwarded to :class:`SearchStrategy`.
    """

    def __init__(
        self,
        initializer: Callable | Initializer,
        operator: Optional[Callable | Operator] = None,
        parent_sel: Optional[Callable | ParentSelection] = None,
        survivor_sel: Optional[Callable | SurvivorSelection] = None,
        name: str = "Strategy from lambda",
        encoding: Optional[Encoding] = None,
        encode_fn: Optional[Callable] = None,
        decode_fn: Optional[Callable] = None,
        parent_selection_amount: Optional[int] = None,
        pop_size: int = 100,
        random_state: Optional[RNGLike] = None,
        **kwargs,
    ):
        if encoding is None and callable(encode_fn) and callable(decode_fn):
            encoding = EncodingFromLambda(encode_fn=encode_fn, decode_fn=decode_fn)

        if callable(initializer):
            initializer = InitializerFromLambda(initializer, pop_size=pop_size, encoding=encoding, random_state=random_state)

        if callable(parent_sel):
            if parent_selection_amount is None:
                parent_selection_amount = initializer.population_size
            parent_sel = ParentSelectionFromLambda(selection_fn=parent_sel, amount=parent_selection_amount, random_state=random_state)

        if callable(operator):
            operator = OperatorFromLambda(operator_fn=operator, encoding=encoding, random_state=random_state)

        if callable(survivor_sel):
            survivor_sel = SurvivorSelectionFromLambda(selection_fn=survivor_sel, random_state=random_state)

        super().__init__(
            initializer=initializer,
            operator=operator,
            parent_sel=parent_sel,
            survivor_sel=survivor_sel,
            name=name,
            random_state=random_state,
            **kwargs,
        )
