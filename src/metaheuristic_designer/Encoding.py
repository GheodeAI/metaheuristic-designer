from __future__ import annotations
from abc import ABC, abstractmethod


class Encoding(ABC):
    """
    Abstract Encoding class

    This class transforms between phenotype and genotype.
    """

    def encode(self, phenotype) -> Any:
        """
        Encodes a viable solution to our problem to the encoding used in each individual of the algorithm.

        Parameters
        ----------
        phenotype: Any
            Information that should be encoded.

        Returns
        -------
        genotype: Any
            Encoded information of the phenotype.
        """

    @abstractmethod
    def decode(self, genotype) -> Any:
        """
        Decodes the contents of an individual to a viable solution to our problem.

        Parameters
        ----------
        genotype: Any
            Information that should be decoded.

        Returns
        -------
        phenotype: Any
            Decoded information of the genotype.
        """
