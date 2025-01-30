from __future__ import annotations
from typing import Any, Iterable
from ..Encoding import Encoding


class CompositeEncoding(Encoding):
    """
    Default encoder that uses the genotype directly as a solution.
    """

    def __init__(self, encodings: Iterable[Encoding]):
        self.encodings = encodings

        super().__init__(vectorized=False)

    def encode_func(self, solutions: Any) -> Any:
        encoded = phenotypes
        for encoding in reversed(self.encodings):
            if encoding.vectorized:
                encoded = encoding.encode_func([encoded])[0]
            else:
                encoded = encoding.encode_func(encoded)
        return encoded

    def decode_func(self, indiv: Any) -> Any:
        decoded = indiv
        for encoding in reversed(self.encodings):
            if encoding.vectorized:
                decoded = encoding.decode_func(decoded[None, :])[0]
            else:
                decoded = encoding.decode_func(decoded)
        return encoded

    def encode(self, solutions: Any) -> Any:
        encoded = solutions
        for encoding in reversed(self.encodings):
            encoded = encoding.encode(encoded)
        return encoded

    def decode(self, population: Any) -> Any:
        decoded = population
        for encoding in self.encodings:
            decoded = encoding.decode(decoded)
        return decoded
