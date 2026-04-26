"""
Base class for the Initializer module.

This module implements functions to generate the initial population of the algorithm.
"""

from __future__ import annotations
from typing import Any, Optional, Callable
from abc import ABC, abstractmethod
import numpy as np
from .population import Population
from .encoding import Encoding, DefaultEncoding
from .objective_function import ObjectiveFunc
from .utils import check_random_state, RNGLike


class Initializer(ABC):
    """
    Abstract population initializer class.

    Parameters
    ----------
    pop_size: int, optional
        Number of individuals to be generated.
    encoding: Encoding, optional
        Encoding that will be passed to each individual.
    """

    def __init__(self, pop_size: int = 1, encoding: Optional[Encoding] = None, random_state: Optional[RNGLike] = None):
        """
        Constructor for the Initializer class.
        """

        self.pop_size = pop_size
        if encoding is None:
            encoding = DefaultEncoding()
        self.encoding = encoding
        self.random_state = check_random_state(random_state)

    @abstractmethod
    def generate_random(self) -> Any:
        """
        Generates a random individual.

        Parameters
        ----------
        objfunc: ObjectiveFunc
            Objective function that will be propagated to the individual.

        Returns
        -------
        new_individual: Any
            Newly generated individual.
        """

    def generate_individual(self) -> Any:
        """
        Define how an individual is initialized

        Parameters
        ----------
        objfunc: ObjectiveFunc
            Objective function that will be propagated to the individual.

        Returns
        -------
        new_individual: Any
            Newly generated individual.
        """

        return self.generate_random()

    def generate_population(self, objfunc: ObjectiveFunc, n_individuals: Optional[int] = None) -> Population:
        """
        Generate n_individual Individuals using the generate_individual method.

        Parameters
        ----------
        objfunc: ObjectiveFunc
            Objective function that will be propagated to each individual.
        n_individual: int, optional
            Number of individuals to generate

        Returns
        -------
        generated_population: Population
            Newly generated population.
        """

        if n_individuals is None:
            n_individuals = self.pop_size

        population_matrix = np.asarray([self.generate_individual() for _ in range(n_individuals)])
        return Population(objfunc, genotype_matrix=population_matrix, encoding=self.encoding)


class InitializerFromLambda(Initializer):
    """
    Initializer that generates individuals with vectors following an user-defined distribution.

    Parameters
    ----------
    generator: callable
        Function that samples an user-defined probability distribution to generate individuals.
    pop_size: int, optional
        Number of individuals to be generated.
    encoding: Encoding, optional
        Encoding that will be passed to each individual.
    """

    def __init__(self, generator: Callable, pop_size: int = 1, encoding: Optional[Encoding] = None, random_state: Optional[RNGLike] = None):
        self.generator = generator

        super().__init__(pop_size=pop_size, encoding=encoding, random_state=random_state)

    def generate_random(self) -> Any:
        return self.generator(self.random_state)
