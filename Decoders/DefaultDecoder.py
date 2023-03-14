import time
import numpy as np
from ..BaseDecoder import BaseDecoder

class DefaultDecoder(BaseDecoder):
    """
    Default encoder that uses the input vector as from the individual as the solution
    """
    
    def encode(self, phenotype: np.ndarray) -> np.ndarray:
        return phenotype
    
    def decode(self, genotype: np.ndarray) -> np.ndarray:
        return genotype
