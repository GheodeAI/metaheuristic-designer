from __future__ import annotations
import numpy as np
from ..Encoding import Encoding


class TypeCastEncoding(Encoding):
    """
    Encoder that uses the input vector from the individual as the solution, the individual
    will be represented in one data type and it will be decoded as another data type
    """

    def __init__(self, encoded_dtype=int, decoded_dtype=float):
        self.encoded_dtype = encoded_dtype
        self.decoded_dtype = decoded_dtype

    def encode(self, phenotype: np.ndarray) -> np.ndarray:
        return phenotype.astype(self.encoded_dtype)

    def decode(self, genotype: np.ndarray) -> np.ndarray:
        return genotype.astype(self.decoded_dtype)
