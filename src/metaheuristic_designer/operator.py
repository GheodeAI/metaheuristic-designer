"""
Base class for the Operator module.

This module implements procedures to modify the current solutions so that we explore the search space.
"""

from __future__ import annotations
import inspect
import logging
from copy import copy
from abc import ABC, abstractmethod
from typing import Optional, Callable
from .encoding import Encoding, DefaultEncoding
from .population import Population
from .initializer import Initializer
from .parametrizable_mixin import ParametrizableMixin
from .utils import check_random_state, RNGLike

logger = logging.getLogger(__name__)


class Operator(ParametrizableMixin, ABC):
    """Abstract base for all perturbation operators.

    An :class:`Operator` modifies a population (typically by
    applying mutation, crossover, or a composite of several steps).
    Subclasses must implement :meth:`evolve`.

    Parameters
    ----------
    name : str, optional
        Display name for this operator.
    encoding : Encoding, optional
        Post-processing applied to the genotype matrix after the
        operator runs.  Defaults to :class:`DefaultEncoding`.
    preserves_order : bool, optional
        If ``True``, the operator keeps individuals in the same
        order (useful for one-to-one survivor selection).
        Default ``False``.
    random_state : RNGLike, optional
        Random number generator.
    **kwargs
        Additional keyword arguments stored as schedulable
        parameters.
    """

    _last_id: int = 0

    def __init__(
        self,
        name: Optional[str] = None,
        encoding: Optional[Encoding] = None,
        preserves_order: bool = False,
        random_state: Optional[RNGLike] = None,
        **kwargs,
    ):
        super().__init__()

        self.id = Operator._last_id
        Operator._last_id += 1

        self.name = name

        if encoding is None:
            encoding = DefaultEncoding()
        self.encoding = encoding

        self.preserves_order = preserves_order

        self.random_state = check_random_state(random_state)
        self.store_kwargs(**kwargs)

    def __call__(self, population: Population, initializer: Optional[Initializer] = None) -> Population:
        """Shorthand for :meth:`evolve`."""
        return self.evolve(population, initializer)

    def gather_params(self):
        """Return the current parameter dictionary (thin wrapper around :meth:`get_params`)."""
        return self.get_params()

    @abstractmethod
    def evolve(self, population: Population, initializer: Optional[Initializer] = None) -> Population:
        """
        Evolves an population using a given strategy.

        Parameters
        ----------
        population: Population
            The population that will be used.
        initializer: Initialize, optional
            The population initializer of the algorithm (used for randomly generating individuals).

        Returns
        -------
        new_population: Population
            The modified population.
        """

    def step(self, progress: float = 0):
        """
        Updates the internal parameters.
        """

        super().step(progress)

        self.encoding.step(progress)

    def get_state(self) -> dict:
        """
        Gets the current state of the algorithm as a dictionary.

        Returns
        -------
        state: dict
            The complete state of the operator.
        """

        data = {"class_name": self.__class__.__name__, "name": self.name, "encoding": self.encoding.get_state(), **self.get_params()}

        return data


class NullOperator(Operator):
    """
    Operator class that returns the individual without changes.
    Surprisingly useful.

    Since it's a no-op, it has the `preserves_order` flag set to True.

    Parameters
    ----------
    name: str, optional
        Name that is associated with the operator.
    """

    def __init__(self, name: Optional[str] = None):
        if name is None:
            name = "Nothing"

        super().__init__(name, preserves_order=True)

    def evolve(self, population: Population, *args) -> Population:
        return copy(population)


class OperatorFromLambda(Operator):
    """Operator that wraps a user‑supplied function.

    The function receives a :class:`Population`, an
    :class:`Initializer`, a random state, and any stored keyword
    arguments, and must return a modified :class:`Population`.

    Parameters
    ----------
    operator_fn : callable
        A function ``(population, initializer, random_state, **kwargs) -> Population``.
    name : str, optional
        Display name (defaults to the function's ``__name__``).
    encoding : Encoding, optional
        See :class:`Operator`.
    preserves_order : bool, optional
        See :class:`Operator`.
    random_state : RNGLike, optional
        See :class:`Operator`.
    **kwargs
        Keyword arguments forwarded to :class:`Operator` and also
        passed to *operator_fn* on each call.
    """

    def __init__(
        self,
        operator_fn: Callable,
        name: Optional[str] = None,
        encoding: Optional[Encoding] = None,
        preserves_order: bool = False,
        random_state: Optional[RNGLike] = None,
        **kwargs,
    ):
        self._validate_function(operator_fn)
        if name is None:
            name = operator_fn.__name__ if hasattr(operator_fn, "__name__") else "Custom operator"

        super().__init__(name, encoding=encoding, preserves_order=preserves_order, random_state=random_state, **kwargs)
        self.operator_fn = operator_fn

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
            raise TypeError(
                f"The function should have at least {required_min_count} positional arguments (`population`, `initializer`, `random_state`)."
            )

    def evolve(self, population: Population, initializer: Optional[Initializer] = None) -> Population:
        return self.operator_fn(population, initializer, self.random_state, **self.current_kwargs)
