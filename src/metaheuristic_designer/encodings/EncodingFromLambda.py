from __future__ import annotations
from typing import Any
from ..Encoding import Encoding


class EncodingFromLambda(Encoding):
    """
    Decoder that uses user specified functions.
    """

    def __init__(self, encode_fn: callable, decode_fn: callable):
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn

    def encode(self, phenotypes: Any) -> Any:
        return self.encode_fn(phenotypes)

    def decode(self, genotypes: Any) -> Any:
        return self.encode_fn(genotypes)
