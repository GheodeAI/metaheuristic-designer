"""
Base class for the Search strategy module.

This module implements the procedure applied in each iteration of the algorithm.
"""

from __future__ import annotations
import logging
from typing import Tuple, Any, Optional
from abc import ABC
import numpy
from .parent_selection import ParentSelection, NullParentSelection
from .survivor_selection import SurvivorSelection, NullSurvivorSelection
from .population import Population
from .initializer import Initializer
from .objective_function import ObjectiveFunc
from .operator import Operator, NullOperator
from .parametrizable_mixin import ParametrizableMixin
from .utils import check_random_state, RNGLike

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
        self._initializer = initializer

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

        self._find_operator_attributes()

    def _find_operator_attributes(self):
        """
        Saves the attributes that represent operators or other relevant information
        about the search strategy.
        """

        attr_dict = vars(self).copy()

        self.parent_sel_register = []
        self.operator_register = []
        self.survivor_sel_register = []

        for var_key in attr_dict:
            attr = attr_dict[var_key]

            if attr:
                # We have a parent selection method
                if isinstance(attr, ParentSelection):
                    self.parent_sel_register.append(attr)

                # We have an operator
                if isinstance(attr, Operator):
                    self.operator_register.append(attr)

                # We have a survivor selection method
                if isinstance(attr, SurvivorSelection):
                    self.survivor_sel_register.append(attr)

                # We have a list of operators
                if isinstance(attr, list) and isinstance(attr[0], Operator):
                    self.operator_register += attr

    @property
    def pop_size(self) -> int:
        """
        Gets the amount of individuals in the population.
        """

        return self._initializer.pop_size

    def best_solution(self, decoded: bool = False) -> Tuple[Any, float]:
        """
        Returns the best solution found by the search strategy and its fitness.

        Returns
        -------
        best_solution : Tuple[Any, float]
            A pair of the best individual with its fitness.
        """

        return self.population.best_solution(decoded)

    @property
    def initializer(self) -> Initializer:
        return self._initializer

    @initializer.setter
    def initializer(self, new_initializer: Initializer):
        self._initializer = new_initializer

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

        if self._initializer is None:
            raise ValueError("Initializer not indicated.")

        self.population = self._initializer.generate_population(objfunc)

        return self.population

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

    def get_state(self, show_population: bool = False) -> dict:
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

        data = {"name": self.name, "initializer": type(self.initializer).__name__}

        data["params"] = self.get_params()

        if self.parent_sel_register:
            data["parent_sel"] = [par.get_state() for par in self.parent_sel_register]

        if self.operator_register:
            data["operators"] = [op.get_state() for op in self.operator_register]

        if self.survivor_sel_register:
            data["survivor_sel"] = [surv.get_state() for surv in self.survivor_sel_register]

        if show_population:
            data["population"] = self.population.get_state()

        return data

    def extra_step_info(self):
        """
        Specific information to report relevant to this search strategy each iteration.
        """

    def extra_report(self, show_plots: bool):
        """
        Specific information to display relevant to this search strategy at the end of the algorithm.

        Parameters
        ----------
        show_plots: bool
            Display plots specific to this search strategy.
        """
