import time
import numpy as np
from ..Decoder import Decoder

class MatrixDecoder(Decoder):
    """
    Decoder used to evolve matrices
    """

    def __init__(self, shape):
        self.shape = shape
    
    def encode(self, phenotype: np.ndarray) -> np.ndarray:
        return np.flatten(genotype)
    
    def decode(self, genotype: np.ndarray) -> np.ndarray:
        return np.reshape(phenotype, self.shape)
