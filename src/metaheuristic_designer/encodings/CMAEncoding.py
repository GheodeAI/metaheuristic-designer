from __future__ import annotations
import numpy as np
from ..Encoding import Encoding
from .DefaultEncoding import DefaultEncoding


class CMAEncoding(Encoding):
    """
    Decoder used to implement the CMA-ES algorithm
    """

    def __init__(self, nparams: int, pre_encoding: Encoding = None):
        self.nparams = nparams

        if pre_encoding is None:
            pre_encoding = DefaultDecoder()
        self.pre_encoding = pre_encoding

    def encode(self, phenotype: np.ndarray, param_vec: np.ndarray = None) -> np.ndarray:
        encoded = self.pre_encoding.encode(phenotype)
        if param_vec is None:
            param_vec = np.ones(self.nparams)
        return np.concatenate(encoded, param_vec)

    def decode(self, genotype: np.ndarray) -> np.ndarray:
        decoded = genotype[: -self.nparams]
        return self.pre_encoding.decode(decoded)
