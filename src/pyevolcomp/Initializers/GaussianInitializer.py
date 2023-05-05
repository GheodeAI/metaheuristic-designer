from __future__ import annotations
import numpy as np
import random
from ..Initializer import Initializer
from ..Individual import Individual

class GaussianInitializer(Initializer):
    def __init__(self, genotype_size, g_mean, g_std, pop_size = 1, encoding = None, dtype = float):
        super().__init__(pop_size, encoding)
        
        self.genotype_size = genotype_size

        if type(g_mean) in [list, tuple, np.ndarray]:
            if len(g_mean) != self.init_len:
                raise ValueError(f"If g_mean is a sequence it must be of length {self.genotype_size}.")
            
            self.g_mean = g_mean
        else:
            self.g_mean = np.repeat(g_mean, self.genotype_size)

        if type(g_std) in [list, tuple, np.ndarray]:
            if len(g_std) != self.init_len:
                raise ValueError(f"If g_std is a sequence it must be of length {self.genotype_size}.")
            
            self.g_std = g_std
        else:
            self.g_std = np.repeat(g_std, self.genotype_size)
        
        self.dtype = dtype


class GaussianVectorInitializer(GaussianInitializer):
    def generate_random(self, objfunc):
        new_vector = np.random.normal(self.g_mean, self.g_std, size=self.genotype_size).astype(self.dtype)
        return Individual(objfunc, new_vector, encoding=self.encoding)

    def generate_individual(self, objfunc):
        return self.generate_random(objfunc)


class GaussianListInitializer(GaussianInitializer):
    def generate_random(self, objfunc):
        new_list = [np.random.normal(m, s) for m, s in zip(self.g_mean, self.g_std)]
        return Individual(objfunc, new_list, encoding=self.encoding)
    
    def generate_individual(self, objfunc):
        return self.generate_random(objfunc)