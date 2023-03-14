import time
import numpy as np
from ..BaseDecoder import BaseDecoder

class MatrixDecoder(BaseDecoder):
    """
    Default encoder that uses the input vector as from the individual as the solution
    """

    def __init__(self, shape):
        self.shape = shape
    
    def encode(self, phenotype: np.ndarray) -> np.ndarray:
        return np.flatten(genotype)
    
    def decode(self, genotype: np.ndarray) -> np.ndarray:
        return np.reshape(phenotype, self.shape)
