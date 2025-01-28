from __future__ import annotations
from typing import Any
from ..Encoding import Encoding


class DefaultEncoding(Encoding):
    """
    Default encoder that uses the genotype directly as a solution.
    """

    def __init__(self):
        super().__init__(vectorized=True)

    def encode_func(self, solutions: Any) -> Any:
        return solutions

    def decode_func(self, population: Any) -> Any:
        return population
