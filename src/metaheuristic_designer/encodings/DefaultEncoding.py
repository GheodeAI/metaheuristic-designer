from __future__ import annotations
from typing import Any
from ..Encoding import Encoding


class DefaultEncoding(Encoding):
    """
    Default encoder that uses the genotype directly as a solution.
    """

    def encode(self, phenotypes: Any) -> Any:
        return phenotypes

    def decode(self, genotypes: Any) -> Any:
        return genotypes
