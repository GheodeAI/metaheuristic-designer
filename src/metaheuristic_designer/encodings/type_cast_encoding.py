"""
Encoding that converts between data types (e.g., float ↔ int ↔ bool).
"""

from __future__ import annotations
from typing import Iterable
from ..encoding import Encoding
from ..utils import MatrixLike


class TypeCastEncoding(Encoding):
    """
    Encoding that converts the population to a different NumPy dtype.

    During encoding, the solutions are cast to `encoded_dtype` (the
    type used internally by operators).  During decoding, they are
    cast to `decoded_dtype` (the type expected by the objective
    function).

    Parameters
    ----------
    encoded_dtype : NumPy dtype, optional
        The dtype used for the internal genotype (default ``int``).
    decoded_dtype : NumPy dtype, optional
        The dtype used for the decoded phenotype (default ``float``).
    \\*\\*kwargs
        Forwarded to :class:`Encoding`.
    """

    def __init__(self, encoded_dtype=int, decoded_dtype=float, **kwargs):
        self.encoded_dtype = encoded_dtype
        self.decoded_dtype = decoded_dtype

        super().__init__(decode_as_array=True, **kwargs)

    def encode(self, solutions: Iterable) -> MatrixLike:
        return solutions.astype(self.encoded_dtype)

    def decode(self, population: MatrixLike) -> Iterable:
        return population.astype(self.decoded_dtype)
