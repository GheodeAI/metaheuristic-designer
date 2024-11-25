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

    def __init__(self, vecsize: int, nparams: int, base_encoding: Encoding = None):
        self.vecsize = vecsize
        self.nparams = nparams
        if base_encoding is None:
            base_encoding = DefaultEncoding()
        self.base_encoding = base_encoding

    def encode(self, phenotype: Any, param_vec: np.ndarray = None) -> np.ndarray:
        phenotype_encoded = self.base_encoding.encode(phenotype)
        return np.concatenate([phenotype_encoded, param_vec])

    def decode(self, genotype: np.ndarray) -> np.ndarray:
        return self.base_encoding.decode(genotype[: self.vecsize])

    def decode_param_vec(self, genotype):
        return genotype[self.vecsize :]

    @abstractmethod
    def decode_param(self, genotype) -> dict:
        pass
