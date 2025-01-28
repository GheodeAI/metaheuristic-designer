from __future__ import annotations
from typing import Any
from ..Encoding import Encoding


class EncodingFromLambda(Encoding):
    """
    Decoder that uses user specified functions.
    """

    def __init__(self, encode_fn: callable, decode_fn: callable, vectorized: bool = False):
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn

        super().__init__(vectorized=vectorized)

    def encode_func(self, solutions: Any) -> Any:
        return self.encode_fn(solutions)

    def decode_func(self, population: Any) -> Any:
        return self.encode_fn(population)
