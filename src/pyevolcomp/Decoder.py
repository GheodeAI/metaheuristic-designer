from abc import ABC, abstractmethod


class Decoder(ABC):
    """
    Base class for transforming between phenotype and genotype
    """

    @abstractmethod
    def encode(self, phenotype):
        """
        Encodes a viable solution to our problem to the encoding used in each individual of the algorithm
        """

    @abstractmethod
    def decode(self, genotype):
        """
        Decodes the contents of an individual to a viable solution to our problem
        """
