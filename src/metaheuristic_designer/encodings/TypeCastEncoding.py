from __future__ import annotations
from numpy import ndarray
from ..Encoding import Encoding


class TypeCastEncoding(Encoding):
    """
    Encoder that uses the input vector from the individual as the solution, the individual
    will be represented in one data type and it will be decoded as another data type
    """

    def __init__(self, encoded_dtype=int, decoded_dtype=float):
        self.encoded_dtype = encoded_dtype
        self.decoded_dtype = decoded_dtype

    def encode(self, phenotypes: ndarray) -> ndarray:
        return phenotypes.astype(self.encoded_dtype)

    def decode(self, genotypes: ndarray) -> ndarray:
        return genotypes.astype(self.decoded_dtype)
