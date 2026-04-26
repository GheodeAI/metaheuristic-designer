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
import numpy as np
from .encoding import Encoding, DefaultEncoding
from .population import Population
from .initializer import Initializer
from .parametrizable_mixin import ParametrizableMixin
from .utils import check_random_state, RNGLike

logger = logging.getLogger(__name__)


class Operator(ParametrizableMixin, ABC):
    """
    Abstract Operator class.

    This class modifies the genotype of one individual in order to perform some optimization task.

    Parameters
    ----------
    name: str, optional
        Name that is associated with the operator.
    encoding: Encoding, optional
        Postprocessing to the operator output.
    """

    _last_id: int = 0

    def __init__(self, name: Optional[str] = None, encoding: Optional[Encoding] = None, random_state: Optional[RNGLike] = None, **kwargs):
        """
        Constructor for the Operator class.
        """
        super().__init__()

        self.id = Operator._last_id
        Operator._last_id += 1

        self.param_scheduler = None

        self.name = name

        if encoding is None:
            encoding = DefaultEncoding()
        self.encoding = encoding

        self.random_state = check_random_state(random_state)
        self.store_kwargs(**kwargs)

    def __call__(
        self,
        population: Population,
        initializer: Optional[Initializer] = None,
    ) -> Population:
        """
        A shorthand for calling the 'evolve' method.
        """

        return self.evolve(population, initializer)

    @abstractmethod
    def evolve(
        self,
        population: Population,
        initializer: Optional[Initializer] = None,
    ) -> Population:
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

        data = {"name": self.name}

        data["encoding"] = self.encoding.get_state()

        data["parameters"] = self.get_params()

        # Serialization fails when encoding anonymous functions, we remove the function if available
        if "function" in data["parameters"]:
            data["parameters"].pop("function")

        return data


class NullOperator(Operator):
    """
    Operator class that returns the individual without changes.
    Surprisingly useful.

    Parameters
    ----------
    name: str, optional
        Name that is associated with the operator.
    """

    def __init__(self, name: Optional[str] = None):
        """
        Constructor for the OperatorNull class
        """

        if name is None:
            name = "Nothing"

        super().__init__(name)

    def evolve(self, population: Population, *args) -> Population:
        return copy(population)


class OperatorFromLambda(Operator):
    """
    Operator class that applies a custom operator specified as a function.

    Parameters
    ----------
    fn: Callable
        Function that will be applied when operating on an individual.
    name: str, optional
        Name that is associated with the operator.
    """

    def __init__(
        self, operator_fn: Callable, name: Optional[str] = None, encoding: Optional[Encoding] = None, random_state: Optional[RNGLike] = None, **kwargs
    ):
        """
        Constructor for the OperatorLambda class
        """

        self._validate_function(operator_fn)
        if name is None:
            name = operator_fn.__name__
        super().__init__(name, encoding=encoding, random_state=random_state, **kwargs)
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
            raise TypeError(f"The function should have at least {required_min_count} positional arguments since it is.")

    def evolve(self, population: Population, initializer: Optional[Initializer] = None) -> Population:
        return self.operator_fn(population, initializer, self.random_state, **self.current_kwargs)
