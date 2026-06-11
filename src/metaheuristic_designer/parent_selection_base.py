"""Base class for the Parent Selection module."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Callable
import inspect
import numpy as np
from .population import Population
from .parametrizable_mixin import ParametrizableMixin
from .utils import check_rng, RNGLike


class ParentSelection(ParametrizableMixin, ABC):
    """Abstract base for all parent selection methods.

    A parent selection chooses which individuals from the current
    population will be used to generate offspring.  Subclasses must
    implement :meth:`select`, which returns a new :class:`Population`
    containing only the selected individuals.

    Parameters
    ----------
    name : str, optional
        Display name for this selection method.
    amount : int, optional
        Default number of individuals to select.  Can be overridden
        at call time.
    rng : RNGLike, optional
        Random number generator.
    **kwargs
        Additional keyword arguments stored as schedulable
        parameters.
    """

    def __init__(self, name: Optional[str] = None, amount: Optional[int] = None, rng: Optional[RNGLike] = None, **kwargs):
        super().__init__()

        self.name = name
        self.rng = check_rng(rng)
        self.store_kwargs(amount=amount, **kwargs)

        self.last_selection_idx = None

    def __call__(self, population: Population, amount: Optional[int] = None) -> Population:
        """Shorthand for :meth:`select`."""
        return self.select(population, amount)

    def gather_params(self) -> dict:
        """Return the current parameter dictionary (thin wrapper around :meth:`get_params`)."""
        return self.get_params()

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
        """Return a dictionary with the selection method's configuration.

        Returns
        -------
        dict
            Keys include ``class_name``, ``name``, and all current
            parameters.
        """

        data = {"class_name": self.__class__.__name__, "name": self.name, **self.get_params()}

        return data


class NullParentSelection(ParentSelection):
    """Null parent selection, returns the whole population unchanged.

    This is the identity element: no individuals are filtered out.
    Useful when the algorithm does not require a parent selection
    step (e.g., random search or certain evolution strategies).

    Parameters
    ----------
    name : str, optional
        Display name. Default ``"Nothing"``.
    **kwargs
        Keyword arguments forwarded to :class:`ParentSelection`.
    """

    def __init__(self, name: Optional[str] = "Nothing", **kwargs):
        super().__init__(name, amount=None, **kwargs)

    def select(self, population: Population, amount: Optional[int] = None) -> Population:
        self.last_selection_idx = np.arange(population.population_size)
        return population.take_selection(self.last_selection_idx)


class ParentSelectionFromLambda(ParentSelection):
    """Parent selection that wraps a user-supplied function.

    The function receives the population, the number of individuals
    to select, a random state, and any stored keyword arguments,
    and must return an array of selected indices.

    Parameters
    ----------
    selection_fn : callable
        A function ``(population, amount, rng, **kwargs) -> indices``.
    name : str, optional
        Display name (defaults to the function's ``__name__``).
    amount : int, optional
        Default number of individuals to select.
    rng : RNGLike, optional
        Random number generator.
    **kwargs
        Keyword arguments forwarded to :class:`ParentSelection`.
    """

    def __init__(self, selection_fn: Callable, name: Optional[str] = None, amount: Optional[int] = None, rng: Optional[RNGLike] = None, **kwargs):
        if name is None:
            name = selection_fn.__name__ if hasattr(selection_fn, "__name__") else "Custom parent selection"

        self._validate_function(selection_fn)
        self.selection_fn = selection_fn

        super().__init__(name=name, amount=amount, rng=rng, **kwargs)

    @staticmethod
    def _validate_function(selection_fn: Callable):
        operator_sig = inspect.signature(selection_fn)

        count = 0
        for p in operator_sig.parameters.values():
            if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                count += 1
            elif p.kind == inspect.Parameter.VAR_POSITIONAL:
                return

        required_min_count = 2
        if count < required_min_count:
            raise TypeError(f"The function should have at least {required_min_count} positional arguments since it is.")

    def select(self, population: Population, amount: Optional[int] = None) -> Population:
        if amount is None:
            if self.current_kwargs["amount"] is None:
                amount = population.population_size
            else:
                amount = self.current_kwargs["amount"]

        params = self.get_params()
        params.pop("amount", None)

        self.last_selection_idx = self.selection_fn(population, amount, self.rng, **params)
        return population.take_selection(self.last_selection_idx)
