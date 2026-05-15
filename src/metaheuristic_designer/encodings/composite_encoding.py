"""
Encoding that chains a sequence of encodings into a single composite operation.
"""

from __future__ import annotations
from typing import Iterable, Optional
from ..encoding import Encoding
from .parameter_extending_encoding import ParameterExtendingEncoding
from ..utils import MatrixLike


class CompositeEncoding(ParameterExtendingEncoding):
    """
    Encoding that applies a sequence of encodings in order.

    Encodings are applied from first to last for decoding, and in reverse
    order for encoding.  This allows stacking transformations such as
    type casting followed by reshaping or sigmoid mapping.

    Parameters
    ----------
    encodings : iterable of Encoding
        The encodings to apply in sequence.
    **kwargs
        Forwarded to :class:`ParameterExtendingEncoding`.
    """

    def __init__(self, encodings: Iterable[Encoding], **kwargs):
        self.encodings = encodings
        dimension = None
        param_sizes = []
        for encoding in encodings:
            if isinstance(encoding, ParameterExtendingEncoding):
                param_sizes += encoding.param_sizes
                dimension = encoding.dimension if dimension is None else min(dimension, encoding.dimension)

        super().__init__(dimension=dimension, param_sizes=param_sizes, **kwargs)

    def gather_params(self):
        all_params = self.get_params()
        for enc in self.encodings:
            all_params.update(enc.gather_params())

        return all_params

    def encode_func(self, solution: Iterable, params: Optional[dict] = None) -> MatrixLike:
        encoded = solution
        for encoding in reversed(self.encodings):
            if isinstance(encoding, ParameterExtendingEncoding):
                encoded = encoding.encode_func(encoded, params)
            else:
                encoded = encoding.encode_func(encoded)

        return encoded

    def decode_func(self, solutions: Iterable) -> MatrixLike:
        decoded = solutions
        for encoding in reversed(self.encodings):
            decoded = encoding.decode_func(decoded)
        return decoded

    def encode(self, solutions: Iterable, params: Optional[dict] = None) -> MatrixLike:
        encoded = solutions
        for encoding in reversed(self.encodings):
            if isinstance(encoding, ParameterExtendingEncoding):
                encoded = encoding.encode(encoded, params)
            else:
                encoded = encoding.encode(encoded)
        return encoded

    def decode(self, population: Iterable) -> MatrixLike:
        decoded = population
        for encoding in self.encodings:
            decoded = encoding.decode(decoded)
        return decoded

    def extract_solution(self, population_matrix: MatrixLike) -> MatrixLike:
        result_population = population_matrix
        for encoding in self.encodings:
            if isinstance(encoding, ParameterExtendingEncoding):
                result_population = encoding.extract_solution(result_population)

        return result_population

    def extract_params(self, population_matrix: MatrixLike) -> MatrixLike:
        result_params = population_matrix
        for encoding in self.encodings:
            if isinstance(encoding, ParameterExtendingEncoding):
                result_params = encoding.extract_params(result_params)

        return result_params
