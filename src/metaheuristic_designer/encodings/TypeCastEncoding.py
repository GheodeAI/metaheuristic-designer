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

        super().__init__(vectorized=True, decode_as_array=True)

    def encode_func(self, solutions: ndarray) -> ndarray:
        return phenotypes.astype(self.encoded_dtype)

    def decode_func(self, population: ndarray) -> ndarray:
        return genotypes.astype(self.decoded_dtype)
