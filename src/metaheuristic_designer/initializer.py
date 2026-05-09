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
    dimension : int
        The dimensionality of the search space (length of individual vectors).
    population_size : int, optional
        Number of individuals to be generated in a population.
    encoding : Encoding, optional
        Encoding that transforms between phenotype and genotype representations.
        If None, defaults to :class:`DefaultEncoding`.
    random_state : RNGLike, optional
        Random state for reproducible individual generation.
    """

    def __init__(self, dimension: int, population_size: int = 1, encoding: Optional[Encoding] = None, random_state: Optional[RNGLike] = None):
        """
        Constructor for the Initializer class.
        """

        self.dimension = dimension
        self.pop_size = population_size
        if encoding is None:
            encoding = DefaultEncoding()
        self.encoding = encoding
        self.random_state = check_random_state(random_state)

    @abstractmethod
    def generate_random(self) -> Any:
        """
        Generates a random individual according to the initializer's distribution.
        
        This abstract method must be implemented by subclasses to define the
        sampling strategy for individual generation. Implementations should use
        :attr:`self.random_state` to ensure reproducibility.

        Returns
        -------
        new_individual : Any
            Newly generated individual in genotype form. The dimensionality should
            match :attr:`self.dimension`.
        
        See Also
        --------
        generate_individual : Wrapper method that typically delegates to this function.
        generate_population : Generates an entire population using this method.
        """

    def generate_individual(self) -> Any:
        """
        Generates a single individual for the population.
        
        By default, this method delegates to :meth:`generate_random`. Subclasses
        can override this to implement custom initialization logic while keeping
        the random sampling strategy separated in :meth:`generate_random`.

        Returns
        -------
        new_individual : Any
            Newly generated individual in genotype form.
        
        See Also
        --------
        generate_random : Abstract method defining the base sampling strategy.
        generate_population : Generates multiple individuals using this method.
        """

        return self.generate_random()

    def generate_population(self, objfunc: ObjectiveFunc, n_individuals: Optional[int] = None) -> Population:
        """
        Generate n_individuals individuals using the generate_individual method.

        Parameters
        ----------
        objfunc : ObjectiveFunc
            Objective function to be optimized by the population.
        n_individuals : int, optional
            Number of individuals to generate. If None, defaults to :attr:`self.pop_size`.

        Returns
        -------
        generated_population : Population
            Newly generated population with fitness calculated based on :obj:`objfunc`.
        
        See Also
        --------
        generate_individual : Generates individual solutions.
        """

        if n_individuals is None:
            n_individuals = self.pop_size

        population_matrix = np.asarray([self.generate_individual() for _ in range(n_individuals)])
        return Population(objfunc, genotype_matrix=population_matrix, encoding=self.encoding)

    def get_state(self):
        """
        Gets the current state of the initializer as a dictionary.
        
        Returns
        -------
        state : dict
            Dictionary containing the class name and other relevant state information.
        """
        data = {"class_name": self.__class__.__name__}

        return data


class InitializerFromLambda(Initializer):
    """
    Initializer that generates individuals using a user-defined function.

    Parameters
    ----------
    generator : Callable
        Function that samples an user-defined probability distribution to generate individuals.
        Should accept a random state as its first argument.
    dimension : int
        The dimensionality of the search space.
    pop_size : int, optional
        Number of individuals to be generated.
    encoding : Encoding, optional
        Encoding that will be passed to each individual.
    random_state : RNGLike, optional
        Random state for reproducible generation.
    """

    def __init__(
        self, generator: Callable, dimension: int, pop_size: int = 1, encoding: Optional[Encoding] = None, random_state: Optional[RNGLike] = None
    ):
        """
        Constructor for the InitializerFromLambda class.
        """
        self.generator = generator

        super().__init__(dimension=dimension, population_size=pop_size, encoding=encoding, random_state=random_state)

    def generate_random(self) -> Any:
        """
        Generates a random individual by calling the user-defined generator function.
        
        Returns
        -------
        new_individual : Any
            Individual generated by the user-provided generator function.
        """
        return self.generator(self.random_state)
