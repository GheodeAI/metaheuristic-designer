from __future__ import annotations
import numpy as np
from ..Initializer import Initializer
from ..utils import RAND_GEN


class PermInitializer(Initializer):
    """
    Initializer that generates individuals with random permutations of n components.

    Parameters
    ----------
    genotype_size: ndarray
        The dimension of the vectors accepted by the objective function.
    pop_size: int, optional
        Number of individuals to be generated.
    """

    def __init__(self, genotype_size, pop_size=1):
        self.genotype_size = genotype_size

        super().__init__(pop_size, encoding=None)

    def generate_random(self):
        return RAND_GEN.permutation(self.genotype_size)
