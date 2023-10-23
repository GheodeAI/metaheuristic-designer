from __future__ import annotations
import numpy as np
from ..Encoding import Encoding


class ImageEncoding(Encoding):
    """
    Decoder used to evolve images
    """

    def __init__(self, shape, color=True):
        if len(shape) == 2:
            shape = tuple(shape)
            if color:
                shape = shape + (3,)
            else:
                shape = shape + (1,)

        self.shape = shape

    def encode(self, phenotype: np.ndarray) -> np.ndarray:
        return np.ndarray.flatten(phenotype)

    def decode(self, genotype: np.ndarray) -> np.ndarray:
        return np.reshape(genotype, self.shape).astype(np.uint8)
