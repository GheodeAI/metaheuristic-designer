from __future__ import annotations
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

    def generate_random(self):
        return self.random_state.permutation(self.dimension)
