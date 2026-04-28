from __future__ import annotations
from typing import Iterable
from numpy import ndarray
from ..encoding import Encoding
from ..utils import MatrixLike


class TypeCastEncoding(Encoding):
    """
    Encoder that uses the input vector from the individual as the solution, the individual
    will be represented in one data type and it will be decoded as another data type
    """

    def __init__(self, encoded_dtype=int, decoded_dtype=float):
        self.encoded_dtype = encoded_dtype
        self.decoded_dtype = decoded_dtype

        super().__init__(decode_as_array=True)

    def encode(self, solutions: Iterable) -> MatrixLike:
        return solutions.astype(self.encoded_dtype)

    def decode(self, population: MatrixLike) -> Iterable:
        return population.astype(self.decoded_dtype)
