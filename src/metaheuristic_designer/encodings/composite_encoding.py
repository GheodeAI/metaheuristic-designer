from __future__ import annotations
from typing import Any, Iterable
from ..encoding import Encoding, ExtendedEncoding


class CompositeEncoding(ExtendedEncoding):
    """
    Default encoder that uses the genotype directly as a solution.
    """

    def __init__(self, encodings: Iterable[Encoding]):
        self.encodings = encodings
        vecsize = None
        param_sizes = []
        for encoding in encodings:
            if isinstance(encoding, ExtendedEncoding):
                param_sizes += encoding.param_sizes
                vecsize = encoding.vecsize if vecsize is None else min(vecsize, encoding.vecsize)

        super().__init__(vecsize=vecsize, param_sizes=param_sizes)

    def encode_func(self, solution: Any) -> Any:
        encoded = solution
        for encoding in reversed(self.encodings):
            if encoding.vectorized:
                encoded = encoding.encode_func([encoded])[0]
            else:
                encoded = encoding.encode_func(encoded)
        return encoded

    def extract_solution(self, population_matrix: ndarray) -> ndarray:
        result_population = population_matrix
        for encoding in self.encodings:
            if isinstance(encoding, ExtendedEncoding):
                result_population = encoding.extract_solution(result_population)

        return result_population

    def extract_params(self, population_matrix: ndarray) -> ndarray:
        result_params = population_matrix
        for encoding in self.encodings:
            if isinstance(encoding, ExtendedEncoding):
                result_params = encoding.extract_params(result_params)

        return result_params

    def decode_func(self, indiv: Any) -> Any:
        decoded = indiv
        for encoding in reversed(self.encodings):
            if encoding.vectorized:
                decoded = encoding.decode_func(decoded[None, :])[0]
            else:
                decoded = encoding.decode_func(decoded)
        return decoded

    def encode(self, solutions: Iterable, params: dict = None) -> ndarray:
        encoded = solutions
        for encoding in reversed(self.encodings):
            if isinstance(encoding, ExtendedEncoding):
                encoded = encoding.encode(encoded, params)
            else:
                encoded = encoding.encode(encoded)
        return encoded

    def decode(self, population: Any) -> Any:
        decoded = population
        for encoding in self.encodings:
            decoded = encoding.decode(decoded)
        return decoded
