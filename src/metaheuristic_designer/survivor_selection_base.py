"""Base class for the Survivor Selection module."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Callable
import inspect
import numpy as np
from .parametrizable_mixin import ParametrizableMixin
from .population import Population
from .utils import check_rng, RNGLike


class SurvivorSelection(ParametrizableMixin, ABC):
    """Abstract base for all survivor selection methods.

    A survivor selection decides which individuals from the current
    population and the newly generated offspring will form the next
    generation.  Subclasses must implement :meth:`select`.

    Parameters
    ----------
    name : str, optional
        Display name for this selection method.
    preserves_order : bool, optional
        If ``True``, the order of individuals is kept
        (useful for one-to-one competition schemes).
        Default ``False``.
    rng : RNGLike, optional
        Random number generator.
    \\*\\*kwargs
        Additional keyword arguments stored as schedulable
        parameters.
    """

    def __init__(self, name: Optional[str] = None, preserves_order: bool = False, rng: Optional[RNGLike] = None, **kwargs):
        super().__init__()

        self.name = name
        self.preserves_order = preserves_order
        self.rng = check_rng(rng)
        self.store_kwargs(**kwargs)

        self.last_selection_idx = None

    def __call__(self, population: Population, offspring: Population) -> Population:
        """Shorthand for :meth:`select`."""

        return self.select(population, offspring)

    def gather_params(self):
        """Return the current parameter dictionary (thin wrapper around :meth:`get_params`)."""
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
            Population containing only the selected survivors.
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


class NullSurvivorSelection(SurvivorSelection):
    """Null survivor selection, offspring replace parents entirely.

    This is the identity element for generational replacement:
    all parents are discarded and all offspring survive.  The
    population size must be maintained by the offspring.

    Parameters
    ----------
    name : str, optional
        Display name. Default ``"Nothing"``.
    \\*\\*kwargs
        Keyword arguments forwarded to :class:`SurvivorSelection`.
    """

    def __init__(self, name: Optional[str] = "Nothing", **kwargs):
        super().__init__(name, preserves_order=True, rng=None, **kwargs)

    def select(self, population: Population, offspring: Population) -> Population:
        self.last_selection_idx = np.arange(population.population_size, population.population_size + offspring.population_size)
        offspring = offspring.update_best_from_parents(population)
        return offspring


class SurvivorSelectionFromLambda(SurvivorSelection):
    """Survivor selection that wraps a user-supplied function.

    The function receives the parent population, the offspring
    population, a random state, and any stored keyword arguments,
    and must return an array of indices into the concatenated
    pool.

    Parameters
    ----------
    selection_fn : callable
        A function ``(parents, offspring, rng, **kwargs) -> indices``.
    name : str, optional
        Display name (defaults to the function's ``__name__``).
    preserves_order : bool, optional
        See :class:`SurvivorSelection`.
    rng : RNGLike, optional
        Random number generator.
    \\*\\*kwargs
        Keyword arguments forwarded to :class:`SurvivorSelection`.
    """

    def __init__(self, selection_fn: Callable, name: Optional[str] = None, preserves_order: bool = False, rng: Optional[RNGLike] = None, **kwargs):
        if name is None:
            name = selection_fn.__name__ if hasattr(selection_fn, "__name__") else "Custom survivor selection"

        self._validate_function(selection_fn)

        self.selection_fn = selection_fn
        super().__init__(name, preserves_order=preserves_order, rng=rng, **kwargs)

    @staticmethod
    def _validate_function(fn: Callable):
        operator_sig = inspect.signature(fn)

        count = 0
        for p in operator_sig.parameters.values():
            if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                count += 1
            elif p.kind == inspect.Parameter.VAR_POSITIONAL:
                return

        required_min_count = 3
        if count < required_min_count:
            raise TypeError(f"The function should have at least {required_min_count} positional arguments (`population`, `offspring`, `rng`).")

    def select(self, population: Population, offspring: Population) -> Population:
        self.last_selection_idx = self.selection_fn(population, offspring, rng=self.rng, **self.current_kwargs)
        return Population.join_populations(population, offspring).take_selection(self.last_selection_idx)
