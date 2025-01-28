from __future__ import annotations
from typing import Any, Iterable
from ..Encoding import Encoding


class CompositeEncoding(Encoding):
    """
    Default encoder that uses the genotype directly as a solution.
    """

    def __init__(self, encodings: Iterable[Encoding]):
        self.encodings = encodings

        super().__init__(vectorized=None)

    def encode_func(self, solutions: Any) -> Any:
        pass

    def decode_func(self, population: Any) -> Any:
        pass

    def encode(self, phenotypes: Any) -> Any:
        encoded = phenotypes
        for encoding in reversed(self.encodings):
            encoded = encoding.encode(encoded)
        return encoded

    def decode(self, genotypes: Any) -> Any:
        decoded = genotypes
        for encoding in self.encodings:
            decoded = encoding.decode(decoded)
        return decoded
