from __future__ import annotations
from typing import Any
from abc import ABC, abstractmethod
import numpy as np
from .Population import Population
from .Encoding import Encoding, DefaultEncoding, ExtendedEncoding
from .ObjectiveFunc import ObjectiveFunc


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

    def __init__(self, pop_size: int = 1, encoding: Encoding = None):
        """
        Constructor for the Initializer class.
        """

        self.pop_size = pop_size
        if encoding is None:
            encoding = DefaultEncoding()
        self.encoding = encoding

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

    def generate_population(self, objfunc: ObjectiveFunc, n_indiv: int = None) -> Population:
        """
        Generate n_indiv Individuals using the generate_individual method.

        Parameters
        ----------
        objfunc: ObjectiveFunc
            Objective function that will be propagated to each individual.
        n_indiv: int, optional
            Number of individuals to generate

        Returns
        -------
        generated_population: Population
            Newly generated population.
        """

        if n_indiv is None:
            n_indiv = self.pop_size

        population_matrix = [self.generate_individual() for _ in range(n_indiv)]
        if isinstance(population_matrix[0], np.ndarray):
            population_matrix = np.asarray(population_matrix)

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

    def __init__(self, generator: callable, pop_size: int = 1, encoding: Encoding = None):
        self.generator = generator

        super().__init__(pop_size, encoding)

    def generate_random(self) -> Any:
        return self.generator()

class ExtendedInitializer(Initializer):
    def __init__(self, solution_init: Initializer, param_init_dict: dict, encoding: ExtendedEncoding):
        assert isinstance(encoding, ExtendedEncoding), "An `ExtendedEncoding` instance must be used with this type of initializer"

        self.solution_init = solution_init
        self.param_init_dict = param_init_dict
        super().__init__(pop_size=self.solution_init.pop_size, encoding=encoding)

    def generate_random(self):
        solution_vector = self.solution_init.generate_random()
        full_vector = np.hstack(
            [solution_vector] +
            [self.param_init_dict[param_name].generate_random() for param_name, _ in self.encoding.param_sizes]
        )
        return full_vector

    def generate_individual(self):
        solution_vector = self.solution_init.generate_individual()
        full_vector = np.hstack(
            [solution_vector] +
            [self.param_init_dict[param_name].generate_individual() for param_name, _ in self.encoding.param_sizes]
        )
        return full_vector