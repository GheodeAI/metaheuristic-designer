from __future__ import annotations
from abc import ABC, abstractmethod
from .Individual import Individual
from .ParamScheduler import ParamScheduler
from .selectionMethods import ParentSelection, SurvivorSelection
from .Operator import Operator


class SearchStrategy(ABC):
    """
    This is the class that defines how the optimization will be carried out.

    Parameters
    ----------
    pop_init: Initializer
        Population initializer that will generate the initial population of the search strategy.
    param: Union[ParamScheduler, dict]
        Dictionary of parameters to define the stopping condition and output of the search strategy.
    name: str, optional
        The name that will be displayed for this search strategy in the reports.
    """

    def __init__(
        self,
        pop_init: Initializer,
        params: Union[ParamScheduler, dict] = None,
        name: str = "some strategy",
    ):
        """
        Constructor of the SearchStrategy class
        """

        self.name = name
        self.pop_init = pop_init

        self.population = None

        self.best = None

        self.param_scheduler = None
        if params is None:
            self.params = {}
        elif isinstance(params, ParamScheduler):
            self.param_scheduler = params
            self.params = self.param_scheduler.get_params()
        else:
            self.params = params

        self._find_operator_attributes()

    def _find_operator_attributes(self):
        """
        Saves the attributes that represent operators or other relevant information
        about the search strategy.
        """

        attr_dict = vars(self).copy()

        self.parent_sel = []
        self.operators = []
        self.surv_sel = []

        for var_key in attr_dict:
            attr = attr_dict[var_key]

            if attr:
                # We have a parent selection method
                if isinstance(attr, ParentSelection):
                    self.parent_sel.append(attr)

                # We have an operator
                if isinstance(attr, Operator):
                    self.operators.append(attr)

                # We have a survivor selection method
                if isinstance(attr, SurvivorSelection):
                    self.surv_sel.append(attr)

                # We have a list of operators
                if isinstance(attr, list) and isinstance(attr[0], Operator):
                    self.operators += attr

    @property
    def pop_size(self):
        """
        Gets the amount of inidividuals in the population.
        """

        return self.pop_init.pop_size

    def best_solution(self) -> Tuple[Individual, float]:
        """
        Returns the best solution found by the search strategy and its fitness.

        Returns
        -------
        best_solution : Tuple[Individual, float]
            A pair of the best individual with its fitness.
        """

        best_fitness = self.best.fitness
        if self.best.objfunc.mode == "min":
            best_fitness *= -1

        return self.best.genotype, best_fitness

    def initialize(self, objfunc: ObjectiveFunc):
        """
        Initializes the optimization search strategy.

        Parameters
        ----------
        objfunc: ObjectiveFunc
            Objective function to be optimized.
        """

        self.population = self.pop_init.generate_population(objfunc)

        self.best = max(self.population, key=lambda x: x.fitness)

    def select_parents(
        self, population: List[Individual], **kwargs
    ) -> Tuple[List[Individual], List[int]]:
        """
        Selects the individuals that will be perturbed in this generation to generate the offspring.

        Parameters
        ----------
        population: List[Individual]
            The current population of the search strategy.

        Returns
        -------
        parents: Tuple[List[Individual], List[int]]
            A pair of the list of individuals considered as parents and their position in the original population.
        """

        return population

    @abstractmethod
    def perturb(
        self, parent_list: List[Individual], objfunc: ObjectiveFunc, **kwargs
    ) -> List[Individual]:
        """
        Applies operators to the population to get the next generation of individuals.

        Parameters
        ----------
        parent_list: List[Individual]
            The current parents that will be used in the search strategy.
        objfunc: ObjectiveFunc
            Objective function to be optimized.

        Returns
        -------
        offspring: List[Individual]
            The list of individuals modified by the operators of the search strategy.
        """

    def select_individuals(
        self, population: List[Individual], offspring: List[Individual], **kwargs
    ) -> List[Individual]:
        """
        Selects the individuals that will pass to the next generation.

        Parameters
        ----------
        population: List[Individual]
            The current population of the search strategy.
        offspring: List[Individual]
            The list of individuals modified by the operators of the search strategy.

        Returns
        -------
        offspring: List[Individual]
            The list of individuals selected for the next generation.
        """

        return offspring

    def update_params(self, **kwargs):
        """
        Updates the parameters of the search strategy and the operators.
        """

    def get_state(self, show_pop: bool = False, show_pop_details: bool = False) -> dict:
        """
        Gets the current state of the search strategy as a dictionary.

        Parameters
        ----------
        show_pop: bool, optional
            Save the current population.
        show_pop_details: bool, optional
            Save the complete details of each individual.

        Returns
        -------
        state: dict
            The complete state of the search strategy.
        """

        data = {
            "name": self.name,
            "population_size": self.pop_size,
        }

        if self.param_scheduler:
            data["param_scheduler"] = self.param_scheduler.get_state()
            data["params"] = self.param_scheduler.get_params()
        elif self.params:
            data["params"] = self.params

        if self.parent_sel:
            data["parent_sel"] = [par.get_state() for par in self.parent_sel]

        if self.operators:
            data["operators"] = [op.get_state() for op in self.operators]

        if self.surv_sel:
            data["survivor_sel"] = [surv.get_state() for surv in self.surv_sel]

        if show_pop:
            data["population"] = [
                ind.get_state(show_speed=show_pop_details, show_best=show_pop_details)
                for ind in self.population
            ]

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
