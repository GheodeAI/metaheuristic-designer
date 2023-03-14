import time
import numpy as np
from ..BaseDecoder import BaseDecoder

class CMADecoder(BaseDecoder):
    """
    Default encoder that uses the input vector as from the individual as the solution
    """

    def __init__(self, vector_size: int, nparams: int):
        self.vector_size = vector_size
        self.nparams = nparams
    
    def encode(self, phenotype: np.ndarray, param_vec: np.ndarray = None) -> np.ndarray:
        if param_vec is None:
            param_vec = np.ones(self.nparams)
        return np.concatenate(phenotype, param_vec)
    
    def decode(self, genotype: np.ndarray) -> np.ndarray:
        return genotype[:vector]
