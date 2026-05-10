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
from .utils import check_random_state, RNGLike, VectorLike


class Initializer(ABC):
    """Abstract base for all population initializers.

    An initializer creates the first generation of individuals.
    It must provide a way to generate a single random genotype
    vector (a 1-D NumPy array) via :meth:`generate_random` and can
    optionally wrap it with a different definition of an individual
    via :meth:`generate_individual`.

    Parameters
    ----------
    dimension : int
        Length of the genotype vector.
    population_size : int, optional
        Number of individuals to generate (default 1).
    encoding : Encoding, optional
        Encoding that will be attached to every individual.
        Defaults to :class:`DefaultEncoding`.
    random_state : RNGLike, optional
        Random number generator.
    """

    def __init__(self, dimension: int, population_size: int = 1, encoding: Optional[Encoding] = None, random_state: Optional[RNGLike] = None):
        self.dimension = dimension
        self.population_size = population_size
        if encoding is None:
            encoding = DefaultEncoding()
        self.encoding = encoding
        self.random_state = check_random_state(random_state)

    @abstractmethod
    def generate_random(self) -> VectorLike:
        """Generate a single random genotype vector (1-D array).

        Returns
        -------
        VectorLike
            A newly generated genotype vector (1-D array).
        """

    def generate_individual(self) -> VectorLike:
        """Generate a single individual.

        By default simply delegates to :meth:`generate_random`.
        Returns a newly generated individual (a 1-D array).
        
        Override this method if your initializer needs to distinguish
        between a randomly initialize individual and a solution
        generated with another strategy (See `SeedProbInitializer`).

        Returns
        -------
        Any
            A newly generated individual.
        """

        return self.generate_random()

    def generate_population(self, objfunc: ObjectiveFunc, n_individuals: Optional[int] = None) -> Population:
        """
        Create a fully formed population of *n_individuals* individuals.

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
            n_individuals = self.population_size

        population_matrix = np.asarray([self.generate_individual() for _ in range(n_individuals)])
        return Population(objfunc, genotype_matrix=population_matrix, encoding=self.encoding)

    def get_state(self) -> dict:
        """Return a minimal dictionary identifying this initializer.

        Returns
        -------
        dict
            Dictionary with key ``"class_name"``.
        """

        data = {"class_name": self.__class__.__name__}

        return data


class InitializerFromLambda(Initializer):
    """Initializer that uses a user-provided function to generate individuals.

    Parameters
    ----------
    generator : callable
        A function ``(random_state) -> genotype`` that returns a
        single genotype vector.
    dimension : int
        Length of the genotype vector.
    pop_size : int, optional
        Number of individuals to generate (default 1).
    encoding : Encoding, optional
        Encoding attached to every individual.
    random_state : RNGLike, optional
        Random number generator.
    """

    def __init__(
        self, generator: Callable, dimension: int, pop_size: int = 1, encoding: Optional[Encoding] = None, random_state: Optional[RNGLike] = None
    ):
        self.generator = generator

        super().__init__(dimension=dimension, population_size=pop_size, encoding=encoding, random_state=random_state)

    def generate_random(self) -> VectorLike:
        return self.generator(self.random_state)
