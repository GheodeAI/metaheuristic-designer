"""
Initializer that generates random permutations.
"""

from __future__ import annotations
from ..initializer import Initializer


class PermInitializer(Initializer):
    """
    Initializer that generates individuals as random permutations of
    integers ``0, 1, …, dimension-1``.

    Parameters
    ----------
    dimension : int
        Length of the permutation (number of elements).
    pop_size : int, optional
        Number of individuals to generate (default 1).
    encoding : Encoding, optional
        Encoding that will be passed to each individual.
    random_state : RNGLike, optional
        Random number generator.
    """

    def generate_random(self):
        return self.random_state.permutation(self.dimension)
