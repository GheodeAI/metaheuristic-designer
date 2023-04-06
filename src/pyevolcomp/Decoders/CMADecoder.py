from __future__ import annotations
import numpy as np
from ..Decoder import Decoder
from .DefaultDecoder import DefaultDecoder


class CMADecoder(Decoder):
    """
    Decoder used to implement the CMA-ES algorithm
    """

    def __init__(self, nparams: int, pre_decoder: Decoder = None):
        self.nparams = nparams
        self.pre_decoder = pre_decoder
        if pre_decoder is None:
            self.pre_decoder = DefaultDecoder()

    def encode(self, phenotype: np.ndarray, param_vec: np.ndarray = None) -> np.ndarray:
        encoded = self.pre_decoder.encode(phenotype)
        if param_vec is None:
            param_vec = np.ones(self.nparams)
        return np.concatenate(encoded, param_vec)

    def decode(self, genotype: np.ndarray) -> np.ndarray:
        decoded = genotype[:-self.nparams]
        return self.pre_decoder.decode(decoded)
