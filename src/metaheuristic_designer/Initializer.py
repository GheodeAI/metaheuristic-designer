from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from .Population import Population
from .encodings import DefaultEncoding


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
        new_individual: Individual
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
        new_individual: Individual
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
        generated_population: List[Individual]
            Newly generated population.
        """

        if n_indiv is None:
            n_indiv = self.pop_size
        
        population_set = [self.generate_individual() for _ in range(n_indiv)]
        if isinstance(population_set[0], np.ndarray):
            population_set = np.asarray(population_set)

        return Population(objfunc, genotype_set=population_set, encoding=self.encoding)
