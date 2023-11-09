from __future__ import annotations
import numpy as np
from ..Encoding import Encoding


class DefaultEncoding(Encoding):
    """
    Default encoder that uses the genotype directly as a solution.
    """

    def encode(self, phenotype: Any) -> Any:
        return phenotype

    def decode(self, genotype: Any) -> Any:
        return genotype
