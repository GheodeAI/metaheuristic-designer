from __future__ import annotations
from typing import Iterable, Optional
import numpy as np
from ..encoding import Encoding
from .parameter_extending_encoding import ParameterExtendingEncoding
from ..utils import MatrixLike


class CompositeEncoding(ParameterExtendingEncoding):
    """
    Default encoder that uses the genotype directly as a solution.
    """

    def __init__(self, encodings: Iterable[Encoding]):
        self.encodings = encodings
        vecsize = None
        param_sizes = []
        for encoding in encodings:
            if isinstance(encoding, ParameterExtendingEncoding):
                param_sizes += encoding.param_sizes
                vecsize = encoding.vecsize if vecsize is None else min(vecsize, encoding.vecsize)

        super().__init__(vecsize=vecsize, param_sizes=param_sizes)

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
