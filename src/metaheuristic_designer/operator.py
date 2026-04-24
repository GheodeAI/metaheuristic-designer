"""
Base class for the Operator module. 

This module implements procedures to modify the current solutions so that we explore the search space.
"""

from __future__ import annotations
import inspect
from copy import copy
from abc import ABC, abstractmethod
import numpy as np
from .encoding import Encoding, DefaultEncoding
from .population import Population
from .initializer import Initializer
from .utils import check_random_state


class Operator(ABC):
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

    _last_id = 0

    def __init__(self, name: str = None, encoding: Encoding = None, random_state=None, **kwargs):
        """
        Constructor for the Operator class.
        """

        self.id = Operator._last_id
        Operator._last_id += 1

        self.param_scheduler = None

        self.name = name

        if encoding is None:
            encoding = DefaultEncoding()
        self.encoding = encoding

        if "method" in kwargs:
            kwargs["method"] = kwargs["method"].lower()

        self.random_state = check_random_state(random_state)

        self.kwargs = kwargs

    def __call__(
        self,
        population: Population,
        initializer: Initializer = None,
    ) -> Population:
        """
        A shorthand for calling the 'evolve' method.
        """

        return self.evolve(population, initializer)

    def step(self, progress: float):
        """
        Updates the parameters of the method using a paramater scheduler if it exists.

        Parameters
        ----------
        progress: float
            Indicator of how close it the algorithm to finishing, 1 means the algorithm should be stopped.
        """

        raise NotImplementedError

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

        data["params"] = copy(self.kwargs)

        # Serialization fails when encoding anonymous functions, we remove the function if available
        if "function" in data["params"]:
            data["params"].pop("function")

        return data

    @abstractmethod
    def evolve(
        self,
        population: Population,
        initializer: Initializer = None,
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


class NullOperator(Operator):
    """
    Operator class that returns the individual without changes.
    Surprisingly useful.

    Parameters
    ----------
    name: str, optional
        Name that is associated with the operator.
    """

    def __init__(self, name: str = None):
        """
        Constructor for the OperatorNull class
        """

        if name is None:
            name = "Nothing"

        super().__init__(name)

    def evolve(self, population, *args):
        return copy(population)


class OperatorFromLambda(Operator):
    """
    Operator class that applies a custom operator specified as a function.

    Parameters
    ----------
    fn: callable
        Function that will be applied when operating on an individual.
    name: str, optional
        Name that is associated with the operator.
    vectorized: bool, optional
        Whether to apply a single operation to the entire population or loop for each individual.
    """

    def __init__(self, operator_fn: callable, name: str = None, encoding: Encoding = None, vectorized = True, random_state=None, **kwargs):
        """
        Constructor for the OperatorLambda class
        """

        self._validate_function(operator_fn, vectorized)

        if name is None:
            name = operator_fn.__name__

        super().__init__(name, encoding=encoding, random_state=random_state, **kwargs)
        self.operator_fn = operator_fn
        self.vectorized = vectorized

    
    @staticmethod
    def _validate_function(operator_fn, vectorized):
        operator_sig = inspect.signature(operator_fn)

        count = 0
        for p in operator_sig.parameters.values():
            if p.kind in inspect.Parameter.POSITIONAL_OR_KEYWORD:
                count += 1
            elif p.kind == inspect.Parameter.VAR_POSITIONAL:
                return
        
        required_min_count = 1 if vectorized else 2
        if count < required_min_count:
            raise TypeError(f"The function should have at least {required_min_count} positional arguments since it is{'' if vectorized else ' not'} vectorized.")

    def evolve(self, population: Population, initializer=None):
        if self.vectorized:
            population = self.operator_fn(population, initializer, self.random_state, **self.kwargs)
        else:
            population = np.asarray([self.operator_fn(copy(indiv), population, initializer, self.random_state, **self.kwargs) for indiv in population.genotype_matrix])

        return population.update_genotype_matrix(population.encode(self.encoding))

