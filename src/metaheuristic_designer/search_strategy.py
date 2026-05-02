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
    """
    Abstract Search Strategy class.

    This is the class that defines how the optimization will be carried out.

    Parameters
    ----------
    initializer: Initializer
        Population initializer that will generate the initial population of the search strategy.
    operator: Operator, optional
        Operator that will be applied to the population each iteration. Defaults to null operator.
    parent_sel: ParentSelection, optional
        Parent selection method that will be applied to the population each iteration. Defaults to returning the entire population.
    survivor_sel: SurvivorSelection, optional
        Survivor selection method that will be applied to the population each iteration. Defaults to a generational selection.
    params: dict, optional
        Dictionary of parameters to define the stopping condition and output of the search strategy.
    name: str, optional
        The name that will be displayed for this search strategy in the reports.
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
        """
        Constructor of the SearchStrategy class
        """

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
    def pop_size(self) -> int:
        """
        Gets the amount of individuals in the population.
        """

        return self.initializer.pop_size

    def best_solution(self, decoded: bool = False) -> Tuple[Any, float]:
        """
        Returns the best solution found by the search strategy and its fitness.

        Returns
        -------
        best_solution : Tuple[Any, float]
            A pair of the best individual with its fitness.
        """

        return self.population.best_solution(decoded)

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

        logger.info("Selected parents...")
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

        logger.info("Applied perturbation operators...")
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

        logger.info("Applied hard constraints...")
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

        logger.info("Selected survivors...")
        return self.survivor_sel(population, offspring)

    def step(self, progress: float):
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
            **self.get_params()
        }

        return data

    def extra_step_info(self):
        """
        Specific information to report relevant to this search strategy each iteration.
        """

    def extra_report(self):
        """
        Specific information to display relevant to this search strategy at the end of the algorithm.

        Parameters
        ----------
        show_plots: bool
            Display plots specific to this search strategy.
        """


class SearchStrategyFromLambda(SearchStrategy):
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
                parent_selection_amount = initializer.pop_size
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
