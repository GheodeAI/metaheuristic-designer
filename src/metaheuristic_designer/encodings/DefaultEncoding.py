from __future__ import annotations
import numpy as np
from ..Encoding import Encoding


class DefaultEncoding(Encoding):
    """
    Default encoder that uses the input vector as from the individual as the solution
    """

    def encode(self, phenotype: Any) -> Any:
        return phenotype

    def decode(self, genotype: Any) -> Any:
        return genotype
