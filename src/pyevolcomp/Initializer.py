from __future__ import annotations
from abc import ABC, abstractmethod
from .Encodings import DefaultEncoding


class Initializer(ABC):
    """
    Abstract population initializer class
    """

    def __init__(self, popSize: int = 1, encoding: Encoding = None):
        self.popSize = popSize
        if encoding is None:
            encoding = DefaultEncoding()
        self.encoding = encoding
    
    @abstractmethod
    def generate_individual(self, objfunc: ObjectiveFunc) -> Individual:
        """
        Define how an individual is initialized
        """
    
    def random_individual(self, objfunc: ObjectiveFunc) -> Individual:
        """
        Generates a random individual
        """

        return self.generate_individual(objfunc)
    
    def generate_population(self, objfunc: ObjectiveFunc, n_indiv: int = None) -> List[Individual]:
        """
        Generate n_indiv Individuals using the generate_individual method.
        """
        
        if n_indiv is None:
            n_indiv = self.popSize
        
        return [self.generate_individual(objfunc) for i in range(n_indiv)]
