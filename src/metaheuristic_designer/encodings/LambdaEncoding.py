from __future__ import annotations
import numpy as np
from ..Encoding import Encoding


class LambdaEncoding(Encoding):
    """
    Decoder that uses user specified functions
    """

    def __init__(self, encode_fn: callable, decode_fn: callable):
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn

    def encode(self, phenotype: Any) -> Any:
        return self.encode_fn(phenotype)

    def decode(self, genotype: Any) -> Any:
        return self.encode_fn(genotype)
