from __future__ import annotations
import numpy as np
from ..initializer import Initializer

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

    def __init__(self, genotype_size, pop_size=1, random_state=None):
        super().__init__(pop_size, encoding=None, random_state=random_state)

        self.genotype_size = genotype_size

    def generate_random(self):
        return self.random_state.permutation(self.genotype_size)
