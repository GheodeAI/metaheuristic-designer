from __future__ import annotations
from typing import Any
from abc import abstractmethod
import numpy as np
from ..Encoding import Encoding
from .DefaultEncoding import DefaultEncoding


class AdaptionEncoding(Encoding):
    """
    Abstract Adaption Encoding class.

    This kind of encoding will represent solutions as a vector with the solution and some parameters for the operators of the algorithm.
    """

    def __init__(self, nparams: int, base_encoding: Encoding = None):
        self.nparams = nparams
        if base_encoding is None:
            base_encoding = DefaultEncoding()
        self.base_encoding = base_encoding

        super().__init__(vectorized=base_encoding.vectorized)

    def encode_func(self, solution: Any, param_vec: np.ndarray = None) -> np.ndarray:
        solution_encoded = self.base_encoding.encode_func(phenotype)
        return np.concatenate([phenotype_encoded, param_vec])

    def encode(self, solutions, param_vec: np.ndarray = None) -> np.ndarray:
        population = None
        if self.vectorized:
            population = self.encode_func(solutions, param_vec)
        else:
            population = np.asarray([self.encode_func(indiv, p) for indiv, p in zip(solutions, param_vec)])

        return population

    def decode_func(self, genotype: np.ndarray) -> np.ndarray:
        return self.base_encoding.decode(genotype[:, : -self.nparams])

    def decode_param_vec(self, genotype):
        return genotype[:, -self.nparams :]

    @abstractmethod
    def decode_param(self, genotype) -> dict:
        pass
