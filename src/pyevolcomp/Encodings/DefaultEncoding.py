from __future__ import annotations
import numpy as np
from ..Encoding import Encoding


class DefaultEncoding(Encoding):
    """
    Default encoder that uses the input vector as from the individual as the solution
    """

    def encode(self, phenotype: np.ndarray) -> np.ndarray:
        return phenotype

    def decode(self, genotype: np.ndarray) -> np.ndarray:
        return genotype
