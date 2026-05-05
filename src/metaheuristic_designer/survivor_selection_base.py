"""
Base class for the Search strategy module.

This module implements the procedure applied in each iteration of the algorithm.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Callable
import inspect
import numpy as np
from .parametrizable_mixin import ParametrizableMixin
from .population import Population
from .utils import check_random_state, RNGLike


class SurvivorSelection(ParametrizableMixin, ABC):
    """
    Abstract Selection Method class.

    This class defines the structure for individual selection methods.

    Parameters
    ----------
    params: dict, optional
        Dictionary of parameters to define the behavior of the selection method.
    name: str, optional
        The name that will be assigned to this selection method.
    """

    def __init__(self, name: Optional[str] = None, preserves_order: bool = False, random_state: Optional[RNGLike] = None, **kwargs):
        """
        Constructor for the SurvivorSelection class
        """
        super().__init__()

        self.name = name
        self.preserves_order = preserves_order
        self.random_state = check_random_state(random_state)
        self.store_kwargs(**kwargs)

        self.last_selection_idx = None

    def __call__(self, population: Population, offspring: Population) -> Population:
        """
        Shorthand for calling the 'select' method
        """

        return self.select(population, offspring)

    def gather_params(self):
        """
        Overridable thin wrapper around get_params
        """

        return self.get_params()

    @abstractmethod
    def select(self, population: Population, offspring: Population) -> Population:
        """
        Takes a population with its offspring and returns the individuals that survive
        to produce the next generation.

        Parameters
        ----------
        population: Population
            Population of individuals that will be selected.
        offspring: Population
            Newly generated individuals to be selected.

        Returns
        -------
        selected: Population
            List of selected individuals.
        """

    def get_state(self):
        """
        Gets the current state of the algorithm as a dictionary.
        """

        data = {"class_name": self.__class__.__name__, "name": self.name, **self.get_params()}

        return data


class NullSurvivorSelection(SurvivorSelection):
    """
    Survivor selection methods.

    Selects the individuals that will remain for the next generation of our algorithm.

    Parameters
    ----------
    method: str
        Strategy used in the selection process.
    params: dict, optional
        Dictionary of parameters to define the behavior of the selection method.
    padding: bool, optional
        Whether to fill the entire list of selected individuals to match the size of the original one.
    name: str, optional
        The name that will be assigned to this selection method.
    """

    def __init__(self, name: Optional[str] = "Nothing", **kwargs):
        """
        Constructor for the SurvivorSelection class
        """

        super().__init__(name, preserves_order=True, random_state=None, **kwargs)

    def select(self, population: Population, offspring: Population) -> Population:
        self.last_selection_idx = np.arange(population.pop_size, population.pop_size + offspring.pop_size)
        offspring = offspring.update_best_from_parents(population)
        return offspring


class SurvivorSelectionFromLambda(SurvivorSelection):
    def __init__(
        self, selection_fn: Callable, name: Optional[str] = None, preserves_order: bool = False, random_state: Optional[RNGLike] = None, **kwargs
    ):
        if name is None:
            name = selection_fn.__name__ if hasattr(selection_fn, "__name__") else "Custom survivor selection"

        self.selection_fn = selection_fn
        super().__init__(name, preserves_order=preserves_order, random_state=random_state, **kwargs)

    @staticmethod
    def _validate_function(operator_fn: Callable):
        operator_sig = inspect.signature(operator_fn)

        count = 0
        for p in operator_sig.parameters.values():
            if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                count += 1
            elif p.kind == inspect.Parameter.VAR_POSITIONAL:
                return

        required_min_count = 3
        if count < required_min_count:
            raise TypeError(f"The function should have at least {required_min_count} positional arguments since it is.")

    def select(self, population: Population, offspring: Population) -> Population:
        selected_idx = self.selection_fn(population, offspring, self.random_state, **self.current_kwargs)
        return Population.join_populations(population, offspring).take_selection(selected_idx)
