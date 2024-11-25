from __future__ import annotations
import numpy as np
from .AdaptionEncoding import AdaptionEncoding


class CMAEncoding(AdaptionEncoding):
    """
    Decoder used to implement the CMA-ES algorithm.
    """

    def decode_param(self, genotype: np.ndarray) -> np.ndarray:
        return {"F": self.decode_param_vec(genotype)}
