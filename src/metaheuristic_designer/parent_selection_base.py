"""
Base class for the Search strategy module.

This module implements the procedure applied in each iteration of the algorithm.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Callable
import inspect
from copy import copy
import numpy as np
from .population import Population
from .parametrizable_mixin import ParametrizableMixin
from .utils import check_random_state, RNGLike


class ParentSelection(ParametrizableMixin, ABC):
    """
    Abstract Parent Selection class.

    This class defines the structure for parent selection methods, deciding
    which solutions are perturbed in the current generation.

    Parameters
    ----------
    params: ParamScheduler or dict, optional
        Dictionary of parameters to define the behavior of the selection method.
    name: str, optional
        The name that will be assigned to this selection method.
    """

    def __init__(self, name: Optional[str] = None, amount: Optional[int] = None, random_state: Optional[RNGLike] = None, **kwargs):
        """
        Constructor for the SurvivorSelection class
        """
        super().__init__()

        self.name = name
        self.random_state = check_random_state(random_state)
        self.store_kwargs(amount=amount, **kwargs)

        self.last_selection_idx = None

    def __call__(self, population: Population, amount: Optional[int] = None) -> Population:
        """
        Shorthand for calling the 'select' method
        """

        return self.select(population, amount)

    @abstractmethod
    def select(self, population: Population, amount: Optional[int] = None) -> Population:
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

    def get_state(self) -> dict:
        """
        Gets the current state of the algorithm as a dictionary.
        """

        data = {"class_name": self.__class__.__name__, "name": self.name, **self.get_params()}

        return data


class NullParentSelection(ParentSelection):
    """
    Parent selection methods.

    Selects the individuals that will be perturbed in this generation.

    Parameters
    ----------
    method: str
        Strategy used in the selection process.
    params: ParamScheduler or dict, optional
        Dictionary of parameters to define the behavior of the selection method.
    padding: bool, optional
        Whether to fill the entire list of selected individuals to match the size of the original one.
    name: str, optional
        The name that will be assigned to this selection method.
    """

    def __init__(self, name: Optional[str] = "Nothing", **kwargs):
        """
        Constructor for the ParentSelection class
        """

        super().__init__(name, amount=None, **kwargs)

    def select(self, population: Population, _amount: Optional[int] = None) -> Population:
        self.last_selection_idx = np.arange(population.pop_size)
        return population.take_selection(self.last_selection_idx)


class ParentSelectionFromLambda(ParentSelection):
    def __init__(
        self, selection_fn: Callable, name: Optional[str] = None, amount: Optional[int] = None, random_state: Optional[RNGLike] = None, **kwargs
    ):
        if name is None:
            name = selection_fn.__name__ if hasattr(selection_fn, "__name__") else "Custom parent selection"

        super().__init__(name=name, amount=amount, random_state=random_state, **kwargs)

        self.selection_fn = selection_fn

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

    def select(self, population: Population, amount: Optional[int] = None) -> Population:
        if amount is None:
            if self.current_kwargs["amount"] is None:
                amount = population.pop_size
            else:
                amount = self.current_kwargs["amount"]

        params = self.get_params()
        params.pop("amount", None)

        self.last_selection_idx = self.selection_fn(population, amount, self.random_state, **params)
        return population.take_selection(self.last_selection_idx)
