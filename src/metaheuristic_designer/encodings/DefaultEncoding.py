from __future__ import annotations
from typing import Any
from ..Encoding import Encoding


class DefaultEncoding(Encoding):
    """
    Default encoder that uses the genotype directly as a solution.
    """

    def __init__(self, decode_as_array=True):
        super().__init__(vectorized=True, decode_as_array=decode_as_array)

    def encode_func(self, solution: Any) -> Any:
        return solution

    def decode_func(self, population: Any) -> Any:
        return population
