from __future__ import annotations
import numpy as np
import random
from ..Initializer import Initializer
from ..Individual import Individual

class UniformInitializer(Initializer):
    def __init__(self, genotype_size, low_lim, up_lim, popSize = 1, encoding = None, dtype = float):
        super().__init__(popSize, encoding)
        
        self.genotype_size = genotype_size

        if type(low_lim) in [list, tuple, np.ndarray]:
            if len(low_lim) != self.init_len:
                raise ValueError(f"If low_lim is a sequence it must be of length {self.genotype_size}.")
            
            self.low_lim = low_lim
        else:
            self.low_lim = np.repeat(low_lim, self.genotype_size)

        if type(up_lim) in [list, tuple, np.ndarray]:
            if len(up_lim) != self.init_len:
                raise ValueError(f"If up_lim is a sequence it must be of length {self.genotype_size}.")
            
            self.up_lim = up_lim
        else:
            self.up_lim = np.repeat(up_lim, self.genotype_size)
        
        self.dtype = dtype


class UniformVectorInitializer(UniformInitializer):    
    def generate_individual(self, objfunc):
        new_vector = np.random.uniform(self.low_lim, self.up_lim, size=self.genotype_size).astype(self.dtype)
        return Individual(objfunc, new_vector, encoding=self.encoding)


class UniformListInitializer(UniformInitializer):
    def generate_individual(self, objfunc):
        new_list = [random.uniform(low, up).astype(self.dtype) for low, up in zip(self.low_lim, self.up_lim)]
        return Individual(objfunc, new_list, encoding=self.encoding)