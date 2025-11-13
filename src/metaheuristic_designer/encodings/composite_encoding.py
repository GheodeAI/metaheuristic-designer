from __future__ import annotations
from typing import Any, Iterable
from ..encoding import Encoding, ExtendedEncoding


class CompositeEncoding(ExtendedEncoding):
    """
    Default encoder that uses the genotype directly as a solution.
    """

    def __init__(self, encodings: Iterable[Encoding]):
        self.encodings = encodings

        # is_extended = any([isinstance(enc, ExtendedEncoding) for enc in encodings])
        super().__init__(vectorized=False, is_extended=is_extended)

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
                result_params = encoding.extract_params(result_population)

        return result_params

    def decode_func(self, indiv: Any) -> Any:
        decoded = indiv
        for encoding in reversed(self.encodings):
            if encoding.vectorized:
                decoded = encoding.decode_func(decoded[None, :])[0]
            else:
                decoded = encoding.decode_func(decoded)
        return decoded

    def encode(self, solutions: Any) -> Any:
        encoded = solutions
        for encoding in reversed(self.encodings):
            encoded = encoding.encode(encoded)
        return encoded

    def decode(self, population: Any) -> Any:
        decoded = population
        for encoding in self.encodings:
            decoded = encoding.decode(decoded)
        return decoded
