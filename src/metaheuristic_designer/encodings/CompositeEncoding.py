from __future__ import annotations
import numpy as np
from ..Encoding import Encoding


class CompositeEncoding(Encoding):
    """
    Default encoder that uses the genotype directly as a solution.
    """

    def __init__(self, encodings: List[Encoding]):
        self.encodings = encodings

    def encode(self, phenotype: Any) -> Any:
        encoded = phenotype
        for encoding in reversed(self.encodings):
            encoded = encoding.encode(encoded)
        return encoded

    def decode(self, genotype: Any) -> Any:
        decoded = genotype
        for encoding in self.encodings:
            decoded = encoding.decode(decoded)
        return decoded
